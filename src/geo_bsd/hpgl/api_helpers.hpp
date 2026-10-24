#ifndef _API_HELPERS_H_KKJJ123NASDLKAS9234ASDnJKNKJNAJKSD
#define _API_HELPERS_H_KKJJ123NASDLKAS9234ASDnJKNKJNAJKSD

struct hpgl_shape_t;
struct hpgl_sgs_params_t;
struct hpgl_ik_params_t;

// F-191: covariance-model range gate shared by the C API boundary. Defined in
// api.cpp so every entry that casts a C `int m_covariance_type` to
// hpgl::covariance_type_t uses ONE predicate and one message text —
// api.cpp (ok/sk/lvm/median-IK/cokriging parameters, mark-I cokriging) and
// api_helpers.cpp (SGS parameters, per-category IK covariances, SIS parameters).
void validate_covariance_type_or_throw(int covariance_type, const char * context);

namespace hpgl
{
	class sugarbox_grid_t;
	class sgs_params_t;
	class ik_params_t;
	struct indicator_params_t;

	int get_shape_volume(hpgl_shape_t * shape);
	void init_grid(hpgl::sugarbox_grid_t & grid, hpgl_shape_t * shape);
	void init_sgs_params(hpgl_sgs_params_t * params, hpgl::sgs_params_t * params2);

	void 
	init_sis_params(
			hpgl_ik_params_t * params, 
			int indicator_count,
			ik_params_t * ikp);

	void
	init_sis_params_2(
		hpgl_ik_params_t * params,
		int indicator_count,
		std::vector<hpgl::indicator_params_t> & result);

	std::string
	get_last_exception_message();

	void
	set_last_exception_message(const char * message);
}


#endif /* _API_HELPERS_H_KKJJ123NASDLKAS9234ASDnJKNKJNAJKSD */
