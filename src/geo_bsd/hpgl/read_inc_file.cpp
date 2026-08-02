#include "stdafx.h"
#include "locale_keeper.h"
#include "hpgl_exception.h"
#include "load_property_from_file.h"
#include <cmath>
#include <climits>
#include <cerrno>
#include <cstring>
#include <cctype>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hpgl
{
	namespace {
		#ifndef _WIN32
		/// Open a file for reading without following a symlink at the final
		/// path component (F-N15). The Python validation layer resolves and
		/// validates the path, then the C++ layer re-opens it by path string —
		/// a plain fopen would follow an attacker-swapped symlink, bypassing
		/// the containment check. Matches the write side (property_writer.cpp)
		/// and the slow parsers (validation.py safe_open_read). Throws
		/// hpgl_exception with a basename-only message on failure.
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

		/// Line-aware token reader matching the Python slow parser semantics
		/// (F-54): only a *line* starting with "/" is the end-of-data marker;
		/// a mid-line "/" token is skipped like any unparseable token. Lines
		/// starting with "--" are skipped whole (bounded to 100KB per line).
		/// Tokens are returned one at a time in line order; returns false at
		/// end-of-data (EOF or a "/" marker line).
		class token_stream_t
		{
		public:
			explicit token_stream_t(FILE * file) : m_file(file), m_pos(0), m_len(0), m_at_end(false) {}

			bool next(char * buffer, size_t buffer_size, const char * func_name)
			{
				for (;;)
				{
					if (m_pos >= m_len)
					{
						if (!fill_line(func_name))
							return false;
					}
					// Skip inter-token whitespace.
					while (m_pos < m_len && isspace(static_cast<unsigned char>(m_line[m_pos])))
						++m_pos;
					if (m_pos >= m_len)
						continue; // blank remainder of line — refill
					size_t start = m_pos;
					while (m_pos < m_len && !isspace(static_cast<unsigned char>(m_line[m_pos])))
						++m_pos;
					size_t tok_len = m_pos - start;
					if (tok_len >= buffer_size)
					{
						// Token longer than the caller's buffer: split it the
						// way fscanf("%Ns") did (hardening, no overflow).
						tok_len = buffer_size - 1;
						m_pos = start + tok_len;
					}
					memcpy(buffer, m_line + start, tok_len);
					buffer[tok_len] = '\0';
					// Comment token anywhere in the line skips the rest of the
					// line (preserves the historical C++ reader behaviour).
					if (tok_len >= 2 && buffer[0] == '-' && buffer[1] == '-')
					{
						m_pos = m_len;
						continue;
					}
					return true;
				}
			}

		private:
			FILE * m_file;
			char m_line[512];
			size_t m_pos;
			size_t m_len;
			bool m_at_end;

			bool fill_line(const char * func_name)
			{
				for (;;)
				{
					if (m_at_end)
						return false;
					if (fgets(m_line, static_cast<int>(sizeof(m_line)), m_file) == nullptr)
					{
						if (ferror(m_file))
							throw hpgl_exception(func_name, "Error reading file.");
						m_at_end = true;
						return false;
					}
					m_len = strlen(m_line);
					m_pos = 0;
					// Line-start comment: skip the whole line (bounded to 100KB).
					if (m_len >= 2 && m_line[0] == '-' && m_line[1] == '-')
					{
						char skip_buf[256];
						size_t total_skipped = m_len;
						const size_t MAX_COMMENT_LINE = 100ULL * 1024ULL;
						while (total_skipped > 0 && m_line[total_skipped - 1] != '\n')
						{
							if (fgets(skip_buf, static_cast<int>(sizeof(skip_buf)), m_file) == nullptr)
								break;
							size_t slen = strlen(skip_buf);
							total_skipped += slen;
							if (total_skipped > MAX_COMMENT_LINE)
								throw hpgl_exception(func_name, "Comment line exceeds 100KB limit.");
							if (slen > 0 && skip_buf[slen - 1] == '\n')
								break;
						}
						continue;
					}
					// Line-start "/": end-of-data marker (matches the Python
					// slow parser, which breaks on lines starting with "/").
					if (m_len >= 1 && m_line[0] == '/')
					{
						m_at_end = true;
						return false;
					}
					return true;
				}
			}
		};

		static void load_floats_into_vector(FILE * file, float * data, int size)
		{
			char buffer[256];
			token_stream_t tokens(file);
			int i = 0;
			while (i < size)
			{
				if (!tokens.next(buffer, static_cast<size_t>(sizeof(buffer)), "load_floats_into_vector"))
					throw hpgl_exception("load_floats_into_vector", "Unexpected end of file.");

				// Mid-line '/' token — the Python slow parser skips it; only a
				// line-start '/' terminates (F-54).
				if (strlen(buffer) >= 1 && buffer[0] == '/')
					continue;

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
				++i;
			}
			// I2-56: validate the token count matches `size`. The slow parser
			// reads every token and _validate_and_reshape_fallback raises on a
			// count mismatch; the fast reader must not silently truncate.
			while (tokens.next(buffer, static_cast<size_t>(sizeof(buffer)), "load_floats_into_vector"))
			{
				if (strlen(buffer) >= 1 && buffer[0] == '/')
					continue; // mid-line '/' is skipped by the Python parser
				std::ostringstream oss;
				oss << "load_floats_into_vector: file contains more than " << size
				    << " values (extra token '" << buffer << "')";
				throw hpgl_exception("load_floats_into_vector", oss.str());
			}
		}

		static void read_bytes(FILE * file,
			int undefined_value,
			unsigned char * data,
			unsigned char * mask,
			int size)
		{
			char buffer[256];
			token_stream_t tokens(file);
			int i = 0;
			while (i < size)
			{
				if (!tokens.next(buffer, static_cast<size_t>(sizeof(buffer)), "read_bytes"))
					throw hpgl_exception("read_bytes", "Unexpected end of file.");

				// Mid-line '/' token — the Python slow parser skips it (F-54).
				if (strlen(buffer) >= 1 && buffer[0] == '/')
					continue;

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
				++i;
			}
			// I2-56: validate the token count matches `size`.
			while (tokens.next(buffer, static_cast<size_t>(sizeof(buffer)), "read_bytes"))
			{
				if (strlen(buffer) >= 1 && buffer[0] == '/')
					continue; // mid-line '/' is skipped by the Python parser
				std::ostringstream oss;
				oss << "read_bytes: file contains more than " << size
				    << " values (extra token '" << buffer << "')";
				throw hpgl_exception("read_bytes", oss.str());
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
		FILE * file = fopen_read_nofollow(file_name, "read_inc_file_float");
		try
		{
			std::string prop_name;
			read_prop_name(file, prop_name);

			load_floats_into_vector(file, data_buffer, size);

			if (mask_buffer != 0)
			{
				// GSLIB missing-value convention (F-M18): values outside the
				// ±1.0e21 window are treated as missing in addition to exact
				// undefined_value matches. The Python slow parsers apply the
				// same window (get_gslib_property / LoadGslibFile), so the fast
				// reader must not load third-party sentinels as data. Strict
				// inequality per the GSLIB convention ("less than -1.0e21 or
				// greater than 1.0e21"); an exact ±1.0e21 value still relies on
				// exact undefined_value equality (float32 round-trip of the
				// HPGL writer's own sentinel is exact).
				const float sentinel_min = -1.0e21f;
				const float sentinel_max =  1.0e21f;
				for (int i = 0; i < size; ++i)
				{
					const float v = data_buffer[i];
					mask_buffer[i] = (v == undefined_value || v < sentinel_min || v > sentinel_max) ? 0 : 1;
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
		FILE * file = fopen_read_nofollow(file_name, "read_inc_file_byte");
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