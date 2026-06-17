#ifndef LOAD_PROPERTY_FROM_FILE_H_INCLUDED
#define LOAD_PROPERTY_FROM_FILE_H_INCLUDED

#include "typedefs.h"
#include "property_array.h"

namespace hpgl
{

// Reads a geostatistical property name from the file (skip comments/blank lines).
// Advances file pointer past the property name line. Throws hpgl_exception on failure.
void read_prop_name(FILE * file, std::string & prop_name);

void load_variable_mean_from_file(
		std::vector<mean_t> & data,
		const std::string & file_name);



}


#endif // LOAD_PROPERTY_FROM_FILE_H_INCLUDED
