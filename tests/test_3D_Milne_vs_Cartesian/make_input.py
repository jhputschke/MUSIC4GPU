import sys
coor, tau0, visc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
L, N = 16.0, 81
zsize = L if coor == "cart" else L/max(tau0, 1e-9)
print(f"""echo_level  1
mode  2
Initial_profile  5
CoorType {1 if coor == 'cart' else 0}
test_blob_e0 10.
test_blob_e_floor 0.001
test_blob_sigma_x 1.0
test_blob_sigma_y 1.5
test_blob_sigma_z 1.0
X_grid_size_in_fm  {L}
Y_grid_size_in_fm  {L}
Eta_grid_size  {zsize:.12g}
Grid_size_in_eta  {N}
Grid_size_in_y  {N}
Grid_size_in_x  {N}
Initial_time_tau_0  {tau0}
Total_evolution_time_tau  3.0
reset_dtau_use_CFL_condition 0
Delta_Tau  0.02
EOS_to_use  0
Viscosity_Flag_Yes_1_No_0  {visc}
Include_Shear_Visc_Yes_1_No_0  {visc}
Shear_to_S_ratio  0.08
T_dependent_Shear_to_S_ratio  0
Include_Bulk_Visc_Yes_1_No_0  {visc}
T_dependent_zeta_over_s 8
Include_second_order_terms {visc}
Include_vorticity_terms 0
turn_on_baryon_diffusion  0
boost_invariant  0
Do_FreezeOut_Yes_1_No_0  0
outputEvolutionData 0
Minmod_Theta 1.8
EndOfData""")
