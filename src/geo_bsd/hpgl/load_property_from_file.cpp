#include "stdafx.h"

#include <cerrno>
#include <cstring>

#include "load_property_from_file.h"
#include "property_array.h"
#include "hpgl_exception.h"
#include "locale_keeper.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hpgl
{

namespace {

#ifndef _WIN32
	/// Open a file for reading without following a symlink at the final
	/// path component (F-N19 / F-N15 parity). This surface has no FFI
	/// wrapper today, but hardening it matches the rest of the read paths
	/// and the write side (property_writer.cpp) in case it is ever exposed.
	FILE * fopen_read_nofollow(const char * file_name, const char * func_name)
	{
		int fd = ::open(file_name, O_RDONLY | O_NOFOLLOW);
		if (fd < 0)
		{
			int open_errno = errno;
			const char * basename = strrchr(file_name, '/');
			if (basename == nullptr) basename = file_name;
			else ++basename; // skip '/'
			std::ostringstream oss;
			oss << "Error opening file '" << basename << "': " << strerror(open_errno);
			throw hpgl_exception(func_name, oss.str());
		}
		FILE * file = fdopen(fd, "r");
		if (file == 0)
		{
			int fdopen_errno = errno;
			close(fd);
			errno = fdopen_errno;
			const char * basename = strrchr(file_name, '/');
			if (basename == nullptr) basename = file_name;
			else ++basename; // skip '/'
			std::ostringstream oss;
			oss << "Error opening file '" << basename << "': " << strerror(fdopen_errno);
			throw hpgl_exception(func_name, oss.str());
		}
		return file;
	}
#else
	// _WIN32: fopen() only — no O_NOFOLLOW available (F-N20 parity
	// documented; junction following is a Windows limitation).
	FILE * fopen_read_nofollow(const char * file_name, const char * func_name)
	{
		FILE * file = fopen(file_name, "r");
		if (file == 0)
		{
			int open_errno = errno;
			const char * basename = strrchr(file_name, '/');
			if (basename == nullptr) basename = file_name;
			else ++basename; // skip '/'
			std::ostringstream oss;
			oss << "Error opening file '" << basename << "': " << strerror(open_errno);
			throw hpgl_exception(func_name, oss.str());
		}
		return file;
	}
#endif

} // anonymous namespace

void read_prop_name(FILE * file, std::string & prop_name)
{
	char line[1024];
start:
	// Use fgets instead of fscanf("%[^\n]") — fscanf returns 0 on blank
	// lines without consuming '\n', causing an infinite goto-start loop.
	// fgets handles blank lines correctly: reads the '\n' and returns a
	// non-null pointer with line_size=0 after stripping.
	if (fgets(line, static_cast<int>(sizeof(line)), file) == nullptr)
	{
		prop_name = "";
		throw hpgl_exception("read_prop_name", "Property name not found.");
	}
	size_t line_size = strlen(line);
	// Strip trailing newline(s)
	if (line_size > 0 && line[line_size - 1] == '\n')
	{
		line[--line_size] = '\0';
		if (line_size > 0 && line[line_size - 1] == '\r')
			line[--line_size] = '\0';
	}
	if (line_size == 0)
		goto start;
	// Accept any non-whitespace-starting line as property name.
	// The writer (property_writer_t) accepts any string — the reader
	// must match. Skip lines starting with whitespace (e.g., blank
	// lines with leading spaces/tabs not caught by the empty check above).
	if (isspace(static_cast<unsigned char>(line[0])))
		goto start;
	// Skip GSLIB-style comment lines beginning with "--"
	// (matching load_doubles_into_vector and read_inc_file.cpp behaviour)
	if (line_size >= 2 && line[0] == '-' && line[1] == '-')
		goto start;
	prop_name = line;
	// Handle continuation for excessively long property names
	// Cap total length to prevent unbounded memory exhaustion (M59 fix)
	const size_t MAX_PROP_NAME_LENGTH = 1024;
	size_t total_len = line_size;
	while (line_size == sizeof(line) - 1 && line[sizeof(line) - 2] != '\n')
	{
		if (fgets(line, static_cast<int>(sizeof(line)), file) == nullptr)
		{
			if (feof(file))
				throw hpgl_exception("read_prop_name",
					"Property name continuation truncated: unexpected end of file.");
			break;
		}
		line_size = strlen(line);
		if (line_size > 0 && line[line_size - 1] == '\n')
			line[--line_size] = '\0';
		total_len += line_size;
		if (total_len > MAX_PROP_NAME_LENGTH)
			throw hpgl_exception("read_prop_name", 
				"Property name exceeds maximum length (1024).");
		prop_name += line;
	}
}

template <typename T>
void load_doubles_into_vector(FILE * file, std::vector<T> & data)
{
	char buffer[256];
	float value;
	int consumed = 0;
	const size_t MAX_ELEMENTS = 100ULL * 1024ULL * 1024ULL; // 100M elements
	while (fscanf(file, "%255s", buffer) == 1)
	{
		// Check for file read errors between reads (M14b safety check)
		if (ferror(file))
			throw hpgl_exception("load_doubles_into_vector", "Error reading file.");

		size_t len = strlen(buffer);
		if (len >= 2 && buffer[0] == '-' && buffer[1] == '-')
		{
			// Bounded comment-line skip (cap at 100KB, M15 fix)
			char skip_buf[256];
			size_t total_skipped = 0;
			const size_t MAX_COMMENT_LINE = 100ULL * 1024ULL;
			while (fgets(skip_buf, static_cast<int>(sizeof(skip_buf)), file))
			{
				size_t slen = strlen(skip_buf);
				total_skipped += slen;
				if (total_skipped > MAX_COMMENT_LINE)
					throw hpgl_exception("load_doubles_into_vector", "Comment line exceeds 100KB limit.");
				if (slen > 0 && skip_buf[slen - 1] == '\n')
					break;
			}
			continue;
		}
		if (len >= 1 && buffer[0] == '/')
		{
			break;
		}

		// Element-count bound (M18 fix)
		if (data.size() >= MAX_ELEMENTS)
			throw hpgl_exception("load_doubles_into_vector",
				"Element count exceeds maximum allowed (100M).");

		// Throw immediately on unparseable tokens — matching
		// load_floats_into_vector behavior in read_inc_file.cpp.
		// E-M73: full-token validation via %n — a bare %f accepts numeric
		// prefixes ("5/" -> 5.0, "1.5abc" -> 1.5) and reports success,
		// silently loading junk (same defect class fixed in
		// read_inc_file.cpp). The %255s token can contain no whitespace,
		// so the whole buffer must be consumed.
		if (sscanf(buffer, "%f%n", &value, &consumed) != 1
			|| consumed != static_cast<int>(len))
		{
			std::ostringstream oss;
			oss << "Error parsing '" << buffer << "' string at token " << data.size();
			throw hpgl_exception("load_doubles_into_vector", oss.str());
		}
		// Reject non-finite values (NaN, Inf) — mirroring
		// load_floats_into_vector in read_inc_file.cpp
		if (!std::isfinite(value))
		{
			std::ostringstream oss;
			oss << "Non-finite value (NaN or Inf) at token " << data.size();
			throw hpgl_exception("load_doubles_into_vector", oss.str());
		}
		data.push_back(value);
	}
	// Check for I/O error after loop exit — fscanf returning < 1 due to
	// a read error is indistinguishable from normal EOF without this.
	if (ferror(file))
		throw hpgl_exception("load_doubles_into_vector", "I/O error reading file.");
}

void load_variable_mean_from_file(
	std::vector<mean_t> & data,
	const std::string & file_name)
{
	blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);
	FILE * file = fopen_read_nofollow(file_name.c_str(), "load_variable_mean_from_file");
	try
	{
		std::string prop_name;
		read_prop_name(file, prop_name);
		load_doubles_into_vector(file, data);
	}
	catch (...)
	{
		fclose(file);
		throw;
	}
	fclose(file);
}

}// hpgl namespace

