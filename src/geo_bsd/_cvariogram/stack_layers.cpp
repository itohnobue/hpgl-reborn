#include "api.h"

#include <math.h>
#include <stdio.h>
#include "stack_layers.h"


CVAR_API void
cvar_stack_layers(
		float_data_t * thick_layers,
		int * layer_markers,
		int layers_count,
		int nz,
		float scalez,
		int blank_value,
		float_data_t * result)
{
	if (thick_layers == nullptr)
	{
		cvar_set_last_error("cvar_stack_layers: thick_layers is null");
		fprintf(stderr,
			"[HPGL ERROR] cvar_stack_layers: thick_layers is null\n");
		fflush(stderr);
		return;
	}
	if (layer_markers == nullptr)
	{
		cvar_set_last_error("cvar_stack_layers: layer_markers is null");
		fprintf(stderr,
			"[HPGL ERROR] cvar_stack_layers: layer_markers is null\n");
		fflush(stderr);
		return;
	}
	if (result == nullptr)
	{
		cvar_set_last_error("cvar_stack_layers: result is null");
		fprintf(stderr,
			"[HPGL ERROR] cvar_stack_layers: result is null\n");
		fflush(stderr);
		return;
	}
	if (layers_count <= 0)
	{
		cvar_set_last_error("cvar_stack_layers: layers_count must be positive");
		fprintf(stderr,
			"[HPGL ERROR] cvar_stack_layers: layers_count must be positive, got %d\n",
			layers_count);
		fflush(stderr);
		return;
	}

	std::vector<float_data_t> layers;
	layers.assign(thick_layers, thick_layers + layers_count);
	
	stack_layers(layers, layer_markers, nz, scalez, blank_value, *result);
}
