date
set_host_options -max_cores 4
set compile_seqmap_propagate_constants     false
set compile_seqmap_propagate_high_effort   false
set compile_enable_register_merging        false
set write_sdc_output_net_resistance        false
set timing_separate_clock_gating_group     true
set verilogout_no_tri tru
set html_log_enable true

set design   aes128_table_ecb
set target_lib   "/home/jb7410/tcbn28hpcplusbwp30p140lvttt0p9v25c_ccs.db /home/jb7410/tcbn28hpcplusbwp30p140hvttt0p9v25c_ccs.db"
set work_dir  [getenv "SYNTH_RUN"]
set verilog_file [getenv "VERILOG_FILE"]

sh mkdir -p $work_dir/snps_reports

set search_path [concat * $search_path]

sh rm -rf $work_dir/snps_work
define_design_lib WORK -path $work_dir/snps_work

set_svf $design.svf

set target_library $target_lib
set link_library $target_lib

analyze -library WORK -format sverilog $verilog_file

elaborate $design
date
link
date

set HVt_lib "tcbn28hpcplusbwp30p140hvttt0p9v25c_ccs"
set LVt_lib "tcbn28hpcplusbwp30p140lvttt0p9v25c_ccs"

set_attribute [get_libs $HVt_lib] default_threshold_voltage_group HVt -type string
set_attribute [get_libs $LVt_lib] default_threshold_voltage_group LVt -type string

write_file -hierarchy -format verilog -output "$work_dir/snps_reports/${design}_flat.v"

report_threshold_voltage_group > $work_dir/snps_reports/report_threshold.rpt

date
exit
