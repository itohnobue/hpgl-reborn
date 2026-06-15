#include "stdafx.h"
#include "locale_keeper.h"
#include "hpgl_exception.h"

namespace hpgl
{
	namespace {
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
			if (!isalpha(static_cast<unsigned char>(line[0])))
			{
				// Skiping line — consume any continuation of a long non-alpha line
				while (line_size == sizeof(line) - 1 && line[sizeof(line) - 2] != '\n')
				{
					if (fgets(line, static_cast<int>(sizeof(line)), file) == nullptr)
						break;
					line_size = strlen(line);
				}
				goto start;
			}
			else
			{
				// Finally line starting with letter
				prop_name = line;
				// Handle continuation for excessively long property names
				while (line_size == sizeof(line) - 1 && line[sizeof(line) - 2] != '\n')
				{
					if (fgets(line, static_cast<int>(sizeof(line)), file) == nullptr)
						break;
					line_size = strlen(line);
					if (line_size > 0 && line[line_size - 1] == '\n')
						line[--line_size] = '\0';
					prop_name += line;
				}
			}
		}

		void load_floats_into_vector(FILE * file, float * data, int size)
		{
			char buffer[256];
			for (int i = 0; i < size; ++i)
			{		
start:
				if (fscanf(file, "%255s", buffer) == EOF)
					throw hpgl_exception("load_floats_into_vector",
						"Unexpected end of file.");
				if (ferror(file))
					throw hpgl_exception("load_floats_into_vector",
						"Error reading file.");
			
				size_t len = strlen(buffer);
				if (len >= 2 && buffer[0] == '-' && buffer[1] == '-')
				{
					//comment - skipping rest of line
					fscanf(file, "%*[^\n]");
					goto start;
				}
				if (len >= 1 && buffer[0] == '/')
				{
					throw hpgl_exception("load_floats_into_vector",
						"Unexpected end of data.");					
				}
				else
				{
					float value;
					if (sscanf(buffer, "%f", &value) != 1)
					{
						std::ostringstream oss;
						oss << "Error parsing '" << buffer << "' string.";
						throw hpgl_exception("load_floats_into_vector", oss.str());
					}					
					data[i] = value;											
				}		
			};
		}		

		void read_bytes(FILE * file, 
			int undefined_value,
			unsigned char * data, 
			unsigned char * mask,			
			int size)
		{
			char buffer[256];
			for (int i = 0; i < size; ++i)
			{		
start:
				if (fscanf(file, "%255s", buffer) == EOF)
					throw hpgl_exception("read_bytes",
						"Unexpected end of file.");
				if (ferror(file))
					throw hpgl_exception("read_bytes",
						"Error reading file.");
			
				size_t len = strlen(buffer);
				if (len >= 2 && buffer[0] == '-' && buffer[1] == '-')
				{
					//comment - skipping rest of line
					fscanf(file, "%*[^\n]");
					goto start;
				}
				if (len >= 1 && buffer[0] == '/')
				{
					throw hpgl_exception("read_bytes",
						"Unexpected end of data.");					
				}
				else
				{
					int value;
					if (sscanf(buffer, "%d", &value) != 1)
					{
						std::ostringstream oss;
						oss << "Error parsing '" << buffer << "' string.";
						throw hpgl_exception("load_floats_into_vector", oss.str());
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
			};
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
			throw hpgl_exception("read_inc_file_float", std::string("Error opening file:") + file_name + ".");
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
		blue_sky::locale_keeper lkeeper ("C", LC_NUMERIC);
		FILE * file = fopen(file_name, "r");
		if (file == 0)
		{
			throw hpgl_exception("read_inc_file_byte", std::string("Error opening file:") + file_name + ".");
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