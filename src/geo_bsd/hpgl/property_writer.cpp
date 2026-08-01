#include "stdafx.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "property_array.h"
#include "property_writer.h"
#include "locale_keeper.h"
#include "hpgl_exception.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hpgl
{
	void property_writer_t :: init(
		const std::string & filename,
		const std::string& property_name)
	{
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
		file_t open_file_checked(const char * filename, const char * mode)
		{
			auto throw_open_error = [filename]() {
				int open_errno = errno;
				// Use basename to avoid leaking full filesystem path in error messages.
				const char * bn = strrchr(filename, '/');
				if (bn == nullptr) bn = filename;
				else ++bn; // skip '/'
				std::ostringstream oss;
				oss << "Can't open file '" << bn << "': " << strerror(open_errno) << ".";
				throw hpgl_exception("open_file_checked", oss.str());
			};
#ifdef _WIN32
			FILE * f = fopen(filename, mode);
			if (f == 0)
				throw_open_error();
#else
			// Open the temp path without following symlinks (I2-58). Plain
			// fopen("w+") follows an attacker-placed symlink at <target>.tmp
			// and writes through it, defeating the Python layer's O_NOFOLLOW
			// path validation. Mirrors validation.py safe_open_write
			// (O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW). Callers always pass "w+",
			// which requires a read-write fd for fdopen compatibility.
			int flags = O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW;
			int fd = open(filename, flags, 0666);
			if (fd < 0)
				throw_open_error();
			FILE * f = fdopen(fd, mode);
			if (f == 0)
			{
				int fdopen_errno = errno;
				close(fd);
				errno = fdopen_errno;
				throw_open_error();
			}
#endif
			return file_t(f, [](FILE* fp) {
				if (fflush(fp) != 0)
					fprintf(stderr, "HPGL: fflush failed — buffered data may be lost\n");
				if (fclose(fp) != 0)
					fprintf(stderr, "HPGL: fclose failed — data may be incomplete\n");
			});
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
			std::string tmp_filename = std::string(filename) + ".tmp";
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				file_t f = open_file_checked(tmp_filename.c_str(), "w+");
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
			// Atomic rename — data only reaches the target path after the
			// complete file is written and flushed. rename() is atomic on
			// macOS/Linux within the same filesystem.
			if (std::rename(tmp_filename.c_str(), filename) != 0)
			{
				int rename_errno = errno;
				std::ostringstream oss;
				oss << "Failed to rename temp file to final: " << strerror(rename_errno);
				throw hpgl_exception("write_property_cont", oss.str());
			}
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
			std::string tmp_filename = std::string(filename) + ".tmp";
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				file_t f = open_file_checked(tmp_filename.c_str(), "w+");
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
			if (std::rename(tmp_filename.c_str(), filename) != 0)
			{
				int rename_errno = errno;
				std::ostringstream oss;
				oss << "Failed to rename temp file to final: " << strerror(rename_errno);
				throw hpgl_exception("write_property_ind", oss.str());
			}
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
			std::string tmp_filename = std::string(filename) + ".tmp";
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				file_t f = open_file_checked(tmp_filename.c_str(), "w+");

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
			if (std::rename(tmp_filename.c_str(), filename) != 0)
			{
				int rename_errno = errno;
				std::ostringstream oss;
				oss << "Failed to rename temp file to final: " << strerror(rename_errno);
				throw hpgl_exception("write_gslib_property_cont_c", oss.str());
			}
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
			std::string tmp_filename = std::string(filename) + ".tmp";
			tmp_file_guard_t tmp_guard(tmp_filename);
			{
				file_t f = open_file_checked(tmp_filename.c_str(), "w+");

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
			if (std::rename(tmp_filename.c_str(), filename) != 0)
			{
				int rename_errno = errno;
				std::ostringstream oss;
				oss << "Failed to rename temp file to final: " << strerror(rename_errno);
				throw hpgl_exception("write_gslib_property_ind_c", oss.str());
			}
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
