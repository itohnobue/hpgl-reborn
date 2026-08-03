#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include "api.h"

// F-08: stack_layers indexes each layer's flat buffer with a map_index built
// from the layer's own strides. Non-contiguous (sliced/strided) views produce
// strides larger than the layer's x/y dims, so map_index can exceed nx*ny and
// write out of bounds into the contiguous cumulative_k buffer. The module
// therefore requires every layer to be C-contiguous with a matching shape.
static bool validate_layer_contiguous(const float_data_t & layer, int nx, int ny)
{
	return layer.m_data != nullptr &&
	       layer.m_data_shape[0] == nx &&
	       layer.m_data_shape[1] == ny &&
	       layer.m_data_strides[1] == 1 &&
	       layer.m_data_strides[0] == ny;
}

void stack_layers(
	std::vector<float_data_t> & thick_layers, 
	int * layers_markers,
	int nz,
	float scalez,
	int blank_value,
	float_data_t & result)
{
	if (thick_layers.empty())
	{
		cvar_set_last_error("stack_layers: no layers provided");
		std::cerr << "[HPGL ERROR] stack_layers: no layers provided" << std::endl;
		return;
	}
	if (scalez <= 0.0f)
	{
		cvar_set_last_error("stack_layers: scalez must be > 0");
		std::cerr << "[HPGL ERROR] stack_layers: scalez must be > 0, got "
		          << scalez << std::endl;
		return;
	}
	if (layers_markers == nullptr)
	{
		cvar_set_last_error("stack_layers: layers_markers is null");
		std::cerr << "[HPGL ERROR] stack_layers: layers_markers is null" << std::endl;
		return;
	}
	if (result.m_data == nullptr)
	{
		cvar_set_last_error("stack_layers: result data is null");
		std::cerr << "[HPGL ERROR] stack_layers: result data is null" << std::endl;
		return;
	}
	if (nz <= 0)
	{
		cvar_set_last_error("stack_layers: nz must be > 0");
		std::cerr << "[HPGL ERROR] stack_layers: nz must be > 0, got "
		          << nz << std::endl;
		return;
	}

	int nx = thick_layers[0].m_data_shape[0];
	int ny = thick_layers[0].m_data_shape[1];

	// F-06: validate the result buffer covers the layer-derived x/y dims.
	// Previously only nz was checked, so a result grid smaller than the layer
	// grid (e.g. layer (10,5) + result (10,4,4)) let cube_index exceed the
	// result volume and wrote out of bounds (heap OOB WRITE / SIGSEGV).
	if (result.m_data_shape[0] < nx ||
	    result.m_data_shape[1] < ny ||
	    result.m_data_shape[2] < nz)
	{
		cvar_set_last_error("stack_layers: result array too small for layer shape");
		std::cerr << "[HPGL ERROR] stack_layers: result array too small for layer shape "
		          << "(result " << result.m_data_shape[0] << "x" << result.m_data_shape[1]
		          << "x" << result.m_data_shape[2] << ", need >= " << nx << "x" << ny
		          << "x" << nz << ")" << std::endl;
		return;
	}

	// F-08: reject non-contiguous (sliced/strided) layer arrays. A strided
	// view would produce map_index beyond nx*ny and corrupt cumulative_k.
	for (size_t layer = 0; layer < thick_layers.size(); ++layer)
	{
		if (!validate_layer_contiguous(thick_layers[layer], nx, ny))
		{
			cvar_set_last_error("stack_layers: layer arrays must be C-contiguous with matching x/y shape");
			std::cerr << "[HPGL ERROR] stack_layers: layer " << layer
			          << " is not C-contiguous or has a mismatched shape" << std::endl;
			return;
		}
	}

	std::vector<double> cumulative_k(nx*ny, 0.0);

	// III-40: initialize the ENTIRE result buffer to blank_value before the
	// layer loop. Previously every cell never touched by a deposit or
	// erosion — notably the top-tail cells above the final surface, and any
	// cell outside the layer x/y footprint — kept whatever the caller
	// pre-filled: a NaN prefill produced a spurious RuntimeError from the
	// Python wrapper's post-call NaN check (cvariogram.py:740-745), and
	// buffer reuse silently corrupted stale cells. Writing a defined value
	// for every cell makes the output independent of the input buffer
	// contents. The deposit/erosion branches below overwrite the cells they
	// own; everything else (top tail, out-of-footprint) stays blank_value.
	for (int z = 0; z < result.m_data_shape[2]; ++z)
		for (int y = 0; y < result.m_data_shape[1]; ++y)
			for (int x = 0; x < result.m_data_shape[0]; ++x)
			{
				int cube_index = result.m_data_strides[0]*x + result.m_data_strides[1]*y + result.m_data_strides[2]*z;
				result.m_data[cube_index] = static_cast<float>(blank_value);
			}

	for(size_t layer = 0; layer < thick_layers.size(); layer++)
	{
		for(int i = 0; i < nx; i++)
		{
			for(int j = 0; j < ny; j++)
			{
				// F-08: after contiguity validation strides[1]==1 and
				// strides[0]==ny, so map_index is always within [0, nx*ny).
				int map_index = thick_layers[layer].m_data_strides[0] * i + thick_layers[layer].m_data_strides[1] * j;

				float thickness = thick_layers[layer].m_data[map_index];

				// I2-25: guard the FP->int casts below. NaN/Inf or huge
				// thickness is UB for static_cast<int>(ceil/floor(...));
				// on x86-64 the NaN cast yields INT_MIN which blanked whole
				// columns and poisoned cumulative_k for every later layer.
				// The bound (1e9 cells) keeps every value safely below
				// INT_MAX (~2.1e9) so the casts are defined; any real model
				// with a billion-cell-thick deposit is absurd.
				if (!std::isfinite(thickness) || std::fabs(static_cast<double>(thickness)) > 1e9)
				{
					cvar_set_last_error("stack_layers: layer thickness must be finite and within range");
					std::cerr << "[HPGL ERROR] stack_layers: layer " << layer
					          << " thickness " << thickness << " is not finite" << std::endl;
					return;
				}

				double new_k = cumulative_k[map_index] + static_cast<double>(thickness) / scalez;
				if (!std::isfinite(new_k) || std::fabs(new_k) > 1e9)
				{
					cvar_set_last_error("stack_layers: cumulative thickness out of range");
					std::cerr << "[HPGL ERROR] stack_layers: cumulative thickness out of range at layer "
					          << layer << std::endl;
					return;
				}

				// positive layer
				if(thickness > 0)
				{
					double old_k = cumulative_k[map_index];

					// F-39: ceil(new_k) instead of floor(new_k)+1. The old
					// +1 filled two cells for an exact-integer deposit
					// (1.0 -> cells 0 and 1); ceil keeps an exact-integer
					// deposit in exactly one cell.
					int k_next = static_cast<int>(ceil(new_k));
					if(k_next > nz)
					{
						k_next = nz;
					}

					// Bounds check: erosion followed by thin deposition can leave
					// cumulative_k negative, producing negative ceil(old_k).
					// Clamp the loop start to 0 (matching negative-path guard at line 61).
					int k_start = static_cast<int>(ceil(old_k));
					if (k_start < 0)
						k_start = 0;
					cumulative_k[map_index] = new_k;
					for(int k = k_start; k < k_next; k++)
						{
							int cube_index = result.m_data_strides[0]*i + result.m_data_strides[1]*j + result.m_data_strides[2]*k;
							result.m_data[cube_index] = static_cast<float>(layers_markers[layer]);
							
						}
				}
				// negative layer
				else if(thickness < 0)
				{
					int k_start = static_cast<int>(ceil(new_k));
					if (k_start < 0)
						k_start = 0;
					for(int k = k_start; k < nz; k++)
					{
						int cube_index = result.m_data_strides[0]*i + result.m_data_strides[1]*j + result.m_data_strides[2]*k;
						result.m_data[cube_index] = static_cast<float>(blank_value);

					}
					cumulative_k[map_index] = new_k;

				}
				// II-57: zero-thickness layer (thickness == 0.0). Previously
				// the else branch treated 0.0 as erosion and blanked the
				// column from ceil(new_k) up — including the ENTIRE column
				// when a zero-thickness layer came first — a silent behavior
				// change vs the same model without the zero layer (realistic
				// in pinch-out / empty-column data). A zero-thickness layer
				// deposits nothing and erodes nothing: skip it, keeping the
				// prior cell values (blank_value or the last layer marker).
				// cumulative_k is unchanged because new_k == old_k here.
				// else: thickness == 0.0 -> no-op
			}
		}
	}
	
}
