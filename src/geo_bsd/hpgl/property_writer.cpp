#include "stdafx.h"

#include "property_array.h"
#include "property_writer.h"
#include "locale_keeper.h"
#include "hpgl_exception.h"

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
		if (fprintf(f, "%E\n", value) < 0)
			throw hpgl_exception("write_value", "Error writing to file.");
	}

	namespace {

		typedef std::shared_ptr<FILE> file_t;
		file_t open_file_checked(const char * filename, const char * mode)
		{
			FILE * f = fopen(filename, mode);
			if (f == 0)
			{
				std::ostringstream oss;
				oss << "Can't open file '" << filename << "'.";
				throw hpgl_exception("open_file_checked", oss.str());
			}
			return file_t(f, [](FILE* fp) {
				if (fclose(fp) != 0)
					fprintf(stderr, "HPGL: fclose failed — data may be incomplete\n");
			});
		}

		void write_property_cont(
				const char * filename,
				const char * property_name,
				const cont_property_array_t & property,
				cont_value_t undefined_value
				)
		{
			blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);
			file_t f = open_file_checked(filename, "w+");
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
			file_t f = open_file_checked(filename, "w+");
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
			file_t f = open_file_checked(filename, "w+");

			int var_num = 1;
			write_header(f.get(), var_num, property_name);

			for (int i = 0, end_i = property.size(); i < end_i; ++i)
			{
				if (property.is_informed(i))
					write_value(f.get(), property.get_at(i));
				else
					write_value(f.get(), undefined_value);
			}
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
			file_t f = open_file_checked(filename, "w+");

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
