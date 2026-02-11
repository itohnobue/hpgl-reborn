/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


/*
 * File:   py_kriging_weights.h
 * Author: nobu
 *
 * Created on 05 March 2009, 11:24
 */

#ifndef _PY_KRIGING_WEIGHTS_H
#define	_PY_KRIGING_WEIGHTS_H

#include "typedefs.h"

namespace hpgl
{
    inline py::list py_calculate_kriging_weight(
        const py::list & center_coords,
        const py::list & neighbourhoods_coords_x,
        const py::list & neighbourhoods_coords_y,
        const py::list & neighbourhoods_coords_z,
            const py_sk_params_t & param)

    {
        assert(py::len(center_coords) == 3);

        assert(py::len(neighbourhoods_coords_x) ==
                py::len(neighbourhoods_coords_y) &&
               py::len(neighbourhoods_coords_y) ==
                py::len(neighbourhoods_coords_z));

        // center point
        real_location_t center(py::cast<double>(center_coords[py::int_(0)]),
                py::cast<double>(center_coords[py::int_(1)]),
                py::cast<double>(center_coords[py::int_(2)]));

        std::vector<real_location_t> neighbourhoods_coords;

        // neighbourhoods
        for (int i = 0; i < py::len(neighbourhoods_coords_x); i++)
        {
            neighbourhoods_coords.push_back(
                    real_location_t(
                        py::cast<double>(neighbourhoods_coords_x[py::int_(i)]),
                        py::cast<double>(neighbourhoods_coords_y[py::int_(i)]),
                        py::cast<double>(neighbourhoods_coords_z[py::int_(i)])
                        )
                    );
        }

        // covariance

        std::vector<kriging_weight_t> weights;
        double variance;

        simple_kriging_weights(
			&param.m_sk_params,
                center,
                neighbourhoods_coords,
                weights,
                variance);


        py::list result_weights;

        for (int i=0; i < (int) weights.size(); i++)
        {
            result_weights.append(double(weights[i]));
        }

        return result_weights;

    }
}



#endif	/* _PY_KRIGING_WEIGHTS_H */

