
! These code/friendly names are generated
! automatically by the script 
! des-pipe/glue/bin/generate_friendly_names.py
! from the file des-pipe/glue/section_names.txt

! If you want to add friendly section names please
! modify that latter file. 
 


module des_section_names
    character(*), parameter :: cosmological_parameters_section = "COSMOPAR"
    character(*), parameter :: halo_model_parameters_section = "HALOPAR"
    character(*), parameter :: intrinsic_alignment_parameters_section = "IA_PARAMS"
    character(*), parameter :: baryon_parameters_section = "BARYON_PAR"
    character(*), parameter :: shear_calibration_parameters_section = "SHEAR_CAL_PAR"
    character(*), parameter :: number_density_params_section = "NZ_PARAMS"
    character(*), parameter :: likelihoods_section = "LIKELIHOODS"
    character(*), parameter :: wl_number_density_section = "NZ_WL"
    character(*), parameter :: matter_power_nl_section = "PK_NL"
    character(*), parameter :: matter_power_lin_section = "PK_LIN"
    character(*), parameter :: shear_xi_section = "SHEAR_XI"
    character(*), parameter :: shear_cl_section = "SHEAR_CL"
    character(*), parameter :: galaxy_cl_section = "GAL_CL"
    character(*), parameter :: cmb_cl_section = "CMB_CL"
    character(*), parameter :: lss_autocorrelation_section = "LSS_ACF"
    character(*), parameter :: gal_matter_power_lin_section = "GAL_PK_LIN"
    character(*), parameter :: linear_cdm_transfer_section = "LIN_TRANSFER"
    character(*), parameter :: sigma_r_lin_section = "SIGMAR_LIN"
    character(*), parameter :: sigma_r_nl_section = "SIGMAR_NL"
    character(*), parameter :: planck_section = "PLANCK"
    character(*), parameter :: intrinsic_alignment_field_section = "IA_FIELD"
    character(*), parameter :: shear_calibration_section = "SHEAR_CAL"
    character(*), parameter :: bias_field_section = "BIAS_FIELD"
    character(*), parameter :: distances_section = "DISTANCES"
    character(*), parameter :: de_equation_of_state_section = "W_OF_A"
end module des_section_names
