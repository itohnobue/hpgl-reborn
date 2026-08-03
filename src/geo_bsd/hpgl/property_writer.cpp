#include "stdafx.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cctype>

#include "property_array.h"
#include "property_writer.h"
#include "locale_keeper.h"
#include "hpgl_exception.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace hpgl
{
	namespace {

		/// Shared property-name validation contract (F-N16). Names written into
		/// INC/GSLIB headers must round-trip through every reader:
		///   - non-empty;
		///   - no control characters (C0 0x00-0x1F or DEL 0x7F) — in particular
		///     no '\n'/'\r', which would inject phantom header lines;
		///   - no leading or trailing whitespace — readers skip
		///     whitespace-leading lines, silently shifting the data off-by-one;
		///   - must not start with "--" (comment marker skipped by readers) or
		///     "/" (end-of-data marker in the INC fast reader).
		/// The Python layer applies the same rule at its call sites.
		void validate_property_name(const std::string & name)
		{
			if (name.empty())
				throw hpgl_exception("validate_property_name", "Property name must not be empty.");
			const char first = name[0];
			if (name.size() >= 2 && first == '-' && name[1] == '-')
				throw hpgl_exception("validate_property_name", "Property name must not start with \"--\".");
			if (first == '/')
				throw hpgl_exception("validate_property_name", "Property name must not start with '/'.");
			if (isspace(static_cast<unsigned char>(first)) ||
			    isspace(static_cast<unsigned char>(name[name.size() - 1])))
				throw hpgl_exception("validate_property_name", "Property name must not have leading or trailing whitespace.");
			for (size_t i = 0; i < name.size(); ++i)
			{
				const unsigned char c = static_cast<unsigned char>(name[i]);
				if (c < 0x20 || c == 0x7f)
					throw hpgl_exception("validate_property_name", "Property name must not contain control characters.");
			}
		}

	} // anonymous namespace

	void property_writer_t :: init(
		const std::string & filename,
		const std::string& property_name)
	{
		// F-N16: reject names that cannot round-trip through every reader
		// (see validate_property_name). Throws hpgl_exception, surfaced as a
		// clean FFI error by the api.cpp callers.
		validate_property_name(property_name);
		m_file_name = filename;
		m_property_name = property_name;
	}

	void write_value(FILE * f, unsigned char value)
	{
		if (fprintf(f, "%d\n", static_cast<int>(value) ) < 0)
			throw hpgl_exception("write_value", "Error writing to file.");
	}

	void write_value(FILE * f, double value)
	{
		// Reject NaN and Inf before writing — these values cannot be
		// read back correctly by parse routines, breaking round-trip.
		if (!std::isfinite(value))
		{
			std::ostringstream oss;
			oss << "Cannot write non-finite value (" << value << ") to file.";
			throw hpgl_exception("write_value", oss.str());
		}
		if (fprintf(f, "%E\n", value) < 0)
			throw hpgl_exception("write_value", "Error writing to file.");
	}

	namespace {

		typedef std::shared_ptr<FILE> file_t;

		[[noreturn]] void throw_open_error(const char * filename)
		{
			int open_errno = errno;
			// Use basename to avoid leaking full filesystem path in error messages.
			const char * bn = strrchr(filename, '/');
			if (bn == nullptr) bn = filename;
			else ++bn; // skip '/'
			std::ostringstream oss;
			oss << "Can't open file '" << bn << "': " << strerror(open_errno) << ".";
			throw hpgl_exception("open_tmp_file_checked", oss.str());
		}

#ifndef _WIN32
		/// Open a uniquely-named temporary file next to `filename` for the
		/// atomic-rename write pattern (F-M19). The name embeds the pid and a
		/// process-local counter, so it cannot be predicted or pre-created by
		/// an attacker with directory write access: the previous deterministic
		/// `<target>.tmp` path allowed a hardlink pre-creation that O_TRUNC
		/// would follow, truncating the linked victim (O_NOFOLLOW blocks
		/// symlinks only), and concurrent writers collided on the fixed name.
		/// O_CREAT|O_EXCL makes the create atomic — a symlink or hardlink
		/// placed at the path is never followed, and a stale temp from a
		/// crashed run (pid reuse) is skipped by retrying the next counter
		/// value. On success `out_path` receives the created path for the RAII
		/// guard and the final rename(). Created with mode 0644 (F-N23: the
		/// previous 0666 & umask left outputs world-readable on permissive
		/// umasks). Callers pass "w+", which requires a read-write fd for
		/// fdopen compatibility.
		file_t open_tmp_file_checked(
				const std::string & filename,
				const std::string & mode,
				std::string & out_path)
		{
			static std::atomic<unsigned long> counter{0};
			const long pid = static_cast<long>(getpid());
			const int max_attempts = 64;
			for (int attempt = 0; attempt < max_attempts; ++attempt)
			{
				std::ostringstream oss;
				oss << filename << ".tmp." << pid << "." << counter.fetch_add(1);
				std::string candidate = oss.str();
				int fd = ::open(candidate.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
				if (fd < 0)
				{
					if (errno == EEXIST)
						continue; // stale temp from a crashed run — try next name
					throw_open_error(candidate.c_str());
				}
				FILE * f = fdopen(fd, mode.c_str());
				if (f == 0)
				{
					int fdopen_errno = errno;
					close(fd);
					std::remove(candidate.c_str()); // don't orphan the temp file
					errno = fdopen_errno;
					throw_open_error(candidate.c_str());
				}
				out_path = candidate;
				return file_t(f, [](FILE* fp) {
					if (fflush(fp) != 0)
						fprintf(stderr, "HPGL: fflush failed — buffered data may be lost\n");
					if (fclose(fp) != 0)
						fprintf(stderr, "HPGL: fclose failed — data may be incomplete\n");
				});
			}
			throw hpgl_exception("open_tmp_file_checked",
				"Could not create a unique temporary file.");
		}
#else
		// _WIN32: fopen() has no O_NOFOLLOW/O_EXCL equivalent, so junction
		// following is a documented limitation (F-N20) — but the temp file must
		// still be a REAL temp file, never the target (II-14). The pre-fix
		// branch opened the TARGET with "w+" (truncating any pre-existing file
		// before a single byte was written), set out_path = filename so the RAII
		// guard was armed on the TARGET, and the final rename was a self-rename
		// no-op — a write error therefore DESTROYED the pre-existing target.
		// Use a unique temp name and exclusive create ("w+x" — the C11 'x'
		// modifier makes fopen fail if the file exists) so the guard can only
		// ever remove the temp.
		file_t open_tmp_file_checked(
				const std::string & filename,
				const std::string & mode,
				std::string & out_path)
		{
			static std::atomic<unsigned long> counter{0};
			const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
			const int max_attempts = 64;
			// Callers pass "w+"; append 'x' for exclusive create ("w+x").
			const std::string xmode = mode + "x";
			for (int attempt = 0; attempt < max_attempts; ++attempt)
			{
				std::ostringstream oss;
				oss << filename << ".tmp." << pid << "." << counter.fetch_add(1);
				std::string candidate = oss.str();
				FILE * f = fopen(candidate.c_str(), xmode.c_str());
				if (f == 0)
				{
					if (errno == EEXIST)
						continue; // stale temp from a crashed run — try next name
					throw_open_error(candidate.c_str());
				}
				out_path = candidate;
				return file_t(f, [](FILE* fp) {
					if (fflush(fp) != 0)
						fprintf(stderr, "HPGL: fflush failed — buffered data may be lost\n");
					if (fclose(fp) != 0)
						fprintf(stderr, "HPGL: fclose failed — data may be incomplete\n");
				});
			}
			throw hpgl_exception("open_tmp_file_checked",
				"Could not create a unique temporary file.");
		}
#endif

		/// Replace `target` with the fully-written `tmp` file (II-14).
		/// On POSIX std::rename atomically replaces the target. On Windows
		/// std::rename FAILS when the target already exists, so MoveFileEx with
		/// MOVEFILE_REPLACE_EXISTING is required to replace an existing file.
		/// The temp only reaches the target path on this success path; any
		/// throw before it leaves the tmp_file_guard to remove only the temp.
		void replace_file(const std::string & tmp, const std::string & target)
		{
#ifdef _WIN32
			if (!MoveFileExA(tmp.c_str(), target.c_str(),
					MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
			{
				DWORD w32_err = GetLastError();
				std::ostringstream oss;
				oss << "Failed to replace file with temp: Windows error " << w32_err;
				throw hpgl_exception("replace_file", oss.str());
			}
#else
			if (std::rename(tmp.c_str(), target.c_str()) != 0)
			{
				int rename_errno = errno;
				std::ostringstream oss;
				oss << "Failed to rename temp file to final: " << strerror(rename_errno);
				throw hpgl_exception("replace_file", oss.str());
			}
#endif
		}

		/// RAII guard that removes the temp file unless the atomic rename
		/// succeeded. Any throw path (write_value failure, fflush failure,
		/// rename failure) previously left <file>.tmp on disk (F-52).
		class tmp_file_guard_t
		{
		public:
			explicit tmp_file_guard_t(const std::string & path) : m_path(path) {}
			~tmp_file_guard_t()
			{
				if (m_armed)
					std::remove(m_path.c_str());
			}
			void disarm() { m_armed = false; }
		private:
			std::string m_path;
			bool m_armed = true;
		};

		void write_property_cont(
				const char * filename,
				const char * property_name,
				const cont_property_array_t & property,
				cont_value_t undefined_value
				)
		{
			blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);

			// Write to a temporary file first, then atomically rename.
			// This prevents data loss if the process crashes mid-write:
			// fopen("w+") would have truncated the original before any
			// data was written, leaving a partial or empty file.
			std::string tmp_filename;
			file_t f = open_tmp_file_checked(filename, "w+", tmp_filename);
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				if (fprintf(f.get(), "%s\n", property_name) < 0)
					throw hpgl_exception("write_property_cont", "Error writing property name.");

				for (int i = 0, end_i = property.size(); i < end_i; ++i)
				{
					if (property.is_informed(i))
						write_value(f.get(), property.get_at(i));
					else
						write_value(f.get(), undefined_value);
				}

				if (fprintf(f.get(), "/\n") < 0)
					throw hpgl_exception("write_property_cont", "Error writing end marker.");

				// Explicit fflush — propagates write failures to the caller
				// instead of silently swallowing them in the shared_ptr deleter.
				if (fflush(f.get()) != 0)
				{
					int flush_errno = errno;
					std::ostringstream oss;
					oss << "fflush failed: " << strerror(flush_errno);
					throw hpgl_exception("write_property_cont", oss.str());
				}
			}
			// Atomic replace — data only reaches the target path after the
			// complete file is written and flushed (rename() on POSIX;
			// MoveFileEx REPLACE_EXISTING on Windows, II-14).
			// R-05: close the temp handle BEFORE the rename. The CRT opens
			// with default _SH_DENYNO sharing (no FILE_SHARE_DELETE), so
			// MoveFileExA on a still-open temp fails with
			// ERROR_SHARING_VIOLATION on EVERY Windows write (pre-fix this
			// path renamed an open file). f.reset() runs the shared_ptr
			// deleter (final fflush + fclose); the explicit fflush above has
			// already propagated write errors to the caller.
			f.reset();
			replace_file(tmp_filename, filename);
			tmp_guard.disarm();
		}

		void write_property_ind(
				const char * filename,
				const char * property_name,
				const indicator_property_array_t & property,
				indicator_value_t undefined_value,
				const std::vector<indicator_value_t> & remap_table
				)
		{
			blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);

			// Write to a temporary file first, then atomically rename
			// (same atomic-write pattern as write_property_cont).
			std::string tmp_filename;
			file_t f = open_tmp_file_checked(filename, "w+", tmp_filename);
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				if (fprintf(f.get(), "%s\n", property_name) < 0)
					throw hpgl_exception("write_property_ind", "Error writing property name.");

				for (int i = 0, end_i = property.size(); i < end_i; ++i)
				{
					if (property.is_informed(i))
					{
						indicator_value_t val = property.get_at(i);
						// Bounds check: invalid data produces undefined_value instead of UB
						if (static_cast<size_t>(val) >= remap_table.size())
							write_value(f.get(), undefined_value);
						else
							write_value(f.get(), remap_table[val]);
					}
					else
						write_value(f.get(), undefined_value);
				}

				if (fprintf(f.get(), "/\n") < 0)
					throw hpgl_exception("write_property_ind", "Error writing end marker.");

				if (fflush(f.get()) != 0)
				{
					int flush_errno = errno;
					std::ostringstream oss;
					oss << "fflush failed: " << strerror(flush_errno);
					throw hpgl_exception("write_property_ind", oss.str());
				}
			}
			// R-05: close the temp handle BEFORE the rename (see
			// write_property_cont). MoveFileExA on a still-open temp fails
			// with ERROR_SHARING_VIOLATION on Windows.
			f.reset();
			replace_file(tmp_filename, filename);
			tmp_guard.disarm();
		}
	}

	namespace {
		void write_header(FILE * f,int var_num, const char * property_name)
		{
			if (fprintf(f, "HPGL saved GSLIB file\n") < 0 ||
			    fprintf(f, "%d\n", var_num) < 0 ||
			    fprintf(f, "%s\n", property_name) < 0)
				throw hpgl_exception("write_header", "Error writing file header.");
		}

		void write_gslib_property_cont_c(
				const char * filename,
				const char * property_name,
				const cont_property_array_t & property,
				cont_value_t undefined_value
				)
		{
			blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);

			// Write to a temporary file first, then atomically rename
			// (same atomic-write pattern as write_property_cont).
			std::string tmp_filename;
			file_t f = open_tmp_file_checked(filename, "w+", tmp_filename);
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				int var_num = 1;
				write_header(f.get(), var_num, property_name);

				for (int i = 0, end_i = property.size(); i < end_i; ++i)
				{
					if (property.is_informed(i))
						write_value(f.get(), property.get_at(i));
					else
						write_value(f.get(), undefined_value);
				}

				if (fflush(f.get()) != 0)
				{
					int flush_errno = errno;
					std::ostringstream oss;
					oss << "fflush failed: " << strerror(flush_errno);
					throw hpgl_exception("write_gslib_property_cont_c", oss.str());
				}
			}
			// R-05: close the temp handle BEFORE the rename (see
			// write_property_cont).
			f.reset();
			replace_file(tmp_filename, filename);
			tmp_guard.disarm();
		}

		void write_gslib_property_ind_c(
				const char * filename,
				const char * property_name,
				const indicator_property_array_t & property,
				indicator_value_t undefined_value,
				const std::vector<indicator_value_t> & remap_table
				)
		{
			blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);

			// Write to a temporary file first, then atomically rename
			// (same atomic-write pattern as write_property_cont).
			std::string tmp_filename;
			file_t f = open_tmp_file_checked(filename, "w+", tmp_filename);
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				int var_num = 1;
				write_header(f.get(), var_num, property_name);

				for (int i = 0, end_i = property.size(); i < end_i; ++i)
				{
					if (property.is_informed(i))
					{
						indicator_value_t val = property.get_at(i);
						// Bounds check: invalid data produces undefined_value instead of UB
						if (static_cast<size_t>(val) >= remap_table.size())
							write_value(f.get(), undefined_value);
						else
							write_value(f.get(), remap_table[val]);
					}
					else
						write_value(f.get(), undefined_value);
				}

				if (fflush(f.get()) != 0)
				{
					int flush_errno = errno;
					std::ostringstream oss;
					oss << "fflush failed: " << strerror(flush_errno);
					throw hpgl_exception("write_gslib_property_ind_c", oss.str());
				}
			}
			// R-05: close the temp handle BEFORE the rename (see
			// write_property_cont).
			f.reset();
			replace_file(tmp_filename, filename);
			tmp_guard.disarm();
		}


	}

	void property_writer_t :: write_double(
			sp_double_property_array_t property,
			double undefined_value)
	{
		write_property_cont(
				m_file_name.c_str(),
				m_property_name.c_str(),
				*property,
				undefined_value);
	}

	void property_writer_t :: write_double(
			const cont_property_array_t & property,
			double undefined_value)
	{
		write_property_cont(
				m_file_name.c_str(),
				m_property_name.c_str(),
				property,
				undefined_value);
	}

	void property_writer_t :: write_byte(
			sp_byte_property_array_t property,
			unsigned char undefined_value,
			const std::vector<unsigned char> & remap_table)
	{
		write_property_ind(
			m_file_name.c_str(),
			m_property_name.c_str(),
			*property,
			undefined_value,
			remap_table);
	}

	void property_writer_t :: write_byte(
			const indicator_property_array_t & property,
			unsigned char undefined_value,
			const std::vector<unsigned char> & remap_table)
	{
		write_property_ind(
			m_file_name.c_str(),
			m_property_name.c_str(),
			property,
			undefined_value,
			remap_table);
	}

		void property_writer_t::write_gslib_double(
			sp_double_property_array_t property,
			double undefined_value)
	{
		write_gslib_property_cont_c(
				m_file_name.c_str(),
				m_property_name.c_str(),
				*property,
				undefined_value);
	}

	void property_writer_t::write_gslib_byte(
			sp_byte_property_array_t property,
			unsigned char undefined_value,
			const std::vector<indicator_value_t> & remap_table)
	{
		write_gslib_property_ind_c(
			m_file_name.c_str(),
			m_property_name.c_str(),
			*property,
			undefined_value,
			remap_table);
	}
}
