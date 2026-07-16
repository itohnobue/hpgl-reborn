#include "stdafx.h"
#include "locale_keeper.h"
#include "hpgl_exception.h"
#include "load_property_from_file.h"
#include <cmath>
#include <climits>
#include <cerrno>
#include <cstring>

namespace hpgl
{
	namespace {
		/// Reads the next non-comment token from the file into buffer.
		/// Skips "--" comment lines (bounded to 100KB per line to prevent
		/// unbounded memory consumption from malicious input).
		/// Returns true on success, false on EOF, throws on file error.
		static bool read_next_token(FILE * file, char * buffer, int buffer_size, const char * func_name)
		{
		start:
			char fmt[16];
			snprintf(fmt, sizeof(fmt), "%%%ds", buffer_size - 1);
			if (fscanf(file, fmt, buffer) == EOF)
				return false;
			if (ferror(file))
				throw hpgl_exception(func_name, "Error reading file.");

			size_t len = strlen(buffer);
			if (len >= 2 && buffer[0] == '-' && buffer[1] == '-')
			{
				// Bounded comment-line skip (cap at 100KB, matching M15 pattern)
				char skip_buf[256];
				size_t total_skipped = 0;
				const size_t MAX_COMMENT_LINE = 100ULL * 1024ULL;
				while (fgets(skip_buf, static_cast<int>(sizeof(skip_buf)), file))
				{
					size_t slen = strlen(skip_buf);
					total_skipped += slen;
					if (total_skipped > MAX_COMMENT_LINE)
						throw hpgl_exception(func_name, "Comment line exceeds 100KB limit.");
					if (slen > 0 && skip_buf[slen - 1] == '\n')
						break;
				}
				goto start;
			}
			return true;
		}

		static void load_floats_into_vector(FILE * file, float * data, int size)
		{
			char buffer[256];
			for (int i = 0; i < size; ++i)
			{
				if (!read_next_token(file, buffer, static_cast<int>(sizeof(buffer)), "load_floats_into_vector"))
					throw hpgl_exception("load_floats_into_vector", "Unexpected end of file.");

				if (strlen(buffer) >= 1 && buffer[0] == '/')
					throw hpgl_exception("load_floats_into_vector", "Unexpected end of data.");

				float value;
				if (sscanf(buffer, "%f", &value) != 1)
				{
					std::ostringstream oss;
					oss << "Error parsing '" << buffer << "' string.";
					throw hpgl_exception("load_floats_into_vector", oss.str());
				}
				if (!std::isfinite(value))
				{
					std::ostringstream oss;
					oss << "Non-finite float value (NaN or Inf) in '"
					    << buffer << "' at position " << i;
					throw hpgl_exception("load_floats_into_vector", oss.str());
				}
				data[i] = value;
			}
		}

		static void read_bytes(FILE * file,
			int undefined_value,
			unsigned char * data,
			unsigned char * mask,
			int size)
		{
			char buffer[256];
			for (int i = 0; i < size; ++i)
			{
				if (!read_next_token(file, buffer, static_cast<int>(sizeof(buffer)), "read_bytes"))
					throw hpgl_exception("read_bytes", "Unexpected end of file.");

				if (strlen(buffer) >= 1 && buffer[0] == '/')
					throw hpgl_exception("read_bytes", "Unexpected end of data.");

				int value;
				if (sscanf(buffer, "%d", &value) != 1)
				{
					std::ostringstream oss;
					oss << "Error parsing '" << buffer << "' string.";
					throw hpgl_exception("read_bytes", oss.str());
				}
				if (value < 0 || value > 255)
				{
					std::ostringstream oss;
					oss << "Byte value " << value << " out of range [0, 255] at position " << i;
					throw hpgl_exception("read_bytes", oss.str());
				}
				data[i] = static_cast<unsigned char>(value);
				mask[i] = value == undefined_value ? 0 : 1;
			}
		}
	}

	void read_inc_file_float(
			const char * file_name,
			float undefined_value,
			int size,
			float * data_buffer,
			unsigned char * mask_buffer)
	{
		blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);
		FILE * file = fopen(file_name, "r");
		if (file == 0)
		{
			// Use basename to avoid leaking full filesystem path in error messages.
			// Also capture the errno before any other call may modify it.
			int open_errno = errno;
			const char * basename = strrchr(file_name, '/');
			if (basename == nullptr) basename = file_name;
			else ++basename; // skip '/'
			std::ostringstream oss;
			oss << "Error opening file '" << basename << "': " << strerror(open_errno);
			throw hpgl_exception("read_inc_file_float", oss.str());
		}
		try
		{
			std::string prop_name;
			read_prop_name(file, prop_name);

			load_floats_into_vector(file, data_buffer, size);

			if (mask_buffer != 0)
			{
				for (int i = 0; i < size; ++i)
				{
					mask_buffer[i] = data_buffer[i] == undefined_value ? 0 : 1;
				}
			}
		}
		catch (...)
		{
			fclose(file);
			throw;
		}
		fclose(file);
	}

	void read_inc_file_byte(
		const char * file_name,
		int undefined_value,
		int size,
		unsigned char * data_buffer,
		unsigned char * mask_buffer)
	{
		// Validate undefined_value fits in unsigned char [0, 255] before
		// processing. The write path (api.cpp:344) enforces this range;
		// the read path must match to prevent out-of-range sentinel values
		// (e.g. -999) from silently marking all cells as informed.
		if (undefined_value < 0 || undefined_value > 255)
		{
			std::ostringstream oss;
			oss << "undefined_value " << undefined_value
			    << " out of range for unsigned char [0, 255]";
			throw hpgl_exception("read_inc_file_byte", oss.str());
		}

		blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);
		FILE * file = fopen(file_name, "r");
		if (file == 0)
		{
			int open_errno = errno;
			const char * basename = strrchr(file_name, '/');
			if (basename == nullptr) basename = file_name;
			else ++basename;
			std::ostringstream oss;
			oss << "Error opening file '" << basename << "': " << strerror(open_errno);
			throw hpgl_exception("read_inc_file_byte", oss.str());
		}
		try
		{
			std::string prop_name;
			read_prop_name(file, prop_name);

			read_bytes(file, undefined_value, data_buffer, mask_buffer, size);
		}
		catch (...)
		{
			fclose(file);
			throw;
		}
		fclose(file);
	}
}