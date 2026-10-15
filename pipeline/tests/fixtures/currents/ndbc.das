Attributes {
    lat {
        String standard_name "latitude";
        String units "degrees_north";
        UInt32 _ChunkSizes 1099;
    }
    lon {
        String standard_name "longitude";
        String units "degrees_east";
        UInt32 _ChunkSizes 700;
    }
    depth {
        String standard_name "depth";
        String units "m";
        String positive "down";
        String axis "Z";
        String bounds "depth_bnds";
        String comment "Nominal depth (and corresponding bounds) based on contributing radars";
    }
    depth_bnds {
        UInt32 _ChunkSizes 2;
    }
    wgs84 {
        String _Unsigned "false";
        String grid_mapping_name "latitude_longitude";
        Float32 longitude_of_prime_meridian 0.0;
        Float32 semi_major_axis 6378137.0;
        Float64 inverse_flattening 298.257223563;
    }
    processing_parameters {
        String _Unsigned "false";
        String long_name "Methods and parameters used to compute total solutions";
        String combine_method_name "Unweighted Least Squares";
        String combine_method_description "Method used to compute total solutions from radial velocities";
        Float32 grid_search_radius 3.0;
        String grid_search_radius_units "km";
        String grid_search_radius_description "Search radius used for finding contributing radial velocities";
        Int32 max_radial_speed 100;
        String max_radial_speed_units "cm s-1";
        String max_radial_speed_description "Maximum radial speed allowed to contribute to total solutions";
        Int32 max_rtv_speed 100;
        String max_rtv_speed_units "cm s-1";
        String max_rtv_speed_description "Maximum allowed total speed";
        Int16 min_radar_sites 2;
        String min_radar_sites_description "Minimum number of radar sites required to make a total solution";
        Int16 min_radials 3;
        String min_radials_description "Minimum number of radials required to make a solution";
        Float32 max_hdop 1.25;
        String max_hdop_description "Maximum allowed HDOP";
    }
    radial_metadata {
        String _Unsigned "false";
        String long_name "Metadata on radial velocities used to compute total solutions";
        Int16 number_files_loaded 42;
        String number_files_loaded_description "Number of radial files loaded";
        String files_loaded "RDLm_ACI1_2026_10_14_1600.ruv, RDLm_ARG1_2026_10_14_1600.ruv, RDLm_BML1_2026_10_14_1600.ruv, RDLm_COP1_2026_10_14_1600.ruv, RDLm_CRIS_2026_10_14_1600.ruv, RDLm_ESTR_2026_10_14_1600.ruv, RDLm_EXPL_2026_10_14_1600.ruv, RDLm_FBK1_2026_10_14_1600.ruv, RDLm_FORT_2026_10_14_1600.ruv, RDLm_GCVE_2026_10_14_1600.ruv, RDLm_GCYN_2026_10_14_1600.ruv, RDLm_LUIS_2026_10_14_1600.ruv, RDLm_MBAC_2026_10_14_1600.ruv, RDLm_MGS1_2026_10_14_1600.ruv, RDLm_MLML_2026_10_14_1600.ruv, RDLm_MONT_2026_10_14_1600.ruv, RDLm_NIC1_2026_10_14_1600.ruv, RDLi_PBON_2026_10_14_1600.ruv, RDLm_PPNS_2026_10_14_1600.ruv, RDLm_PREY_2026_10_14_1600.ruv, RDLm_PTC1_2026_10_14_1600.ruv, RDLm_PTM1_2026_10_14_1600.ruv, RDLm_RFG1_2026_10_14_1600.ruv, RDLm_RTC1_2026_10_14_1600.ruv, RDLm_SAUS_2026_10_14_1600.ruv, RDLm_SCCI_2026_10_14_1600.ruv, RDLi_SCDB_2026_10_14_1600.ruv, RDLm_SCDH_2026_10_14_1600.ruv, RDLm_SCI1_2026_10_14_1600.ruv, RDLi_SCPF_2026_10_14_1600.ruv, RDLm_SCRZ_2026_10_14_1600.ruv, RDLi_SCTB_2026_10_14_1600.ruv, RDLm_SDBP_2026_10_14_1600.ruv, RDLi_SDCP_2026_10_14_1600.ruv, RDLm_SDDP_2026_10_14_1600.ruv, RDLm_SDSE_2026_10_14_1600.ruv, RDLm_SDUT_2026_10_14_1600.ruv, RDLm_SLID_2026_10_14_1600.ruv, RDLm_SNI1_2026_10_14_1600.ruv, RDLm_SSD1_2026_10_14_1600.ruv, RDLi_WSH1_2026_10_14_1600.ruv, RDLi_YHS2_2026_10_14_1600.ruv";
        String files_loaded_description "Radial file names loaded";
    }
    prod_mon {
        String long_name "Total (actual) number of retrievals";
    }
    time {
        String standard_name "time";
        String units "seconds since 1970-01-01";
        String calendar "gregorian";
        String bounds "time_bnds";
        UInt32 _ChunkSizes 1024;
    }
    time_bnds {
        UInt32 _ChunkSizes 1, 2;
    }
    u {
        Int16 _FillValue -32767;
        String standard_name "surface_eastward_sea_water_velocity";
        String units "m s-1";
        Float32 scale_factor 0.01;
        String grid_mapping "wgs84";
        String coordinates "depth";
        String cell_methods "depth: mean time: mean";
        String ancillary_variables "dopx";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    v {
        Int16 _FillValue -32767;
        String standard_name "surface_northward_sea_water_velocity";
        String units "m s-1";
        Float32 scale_factor 0.01;
        String grid_mapping "wgs84";
        String coordinates "depth";
        String cell_methods "depth: mean time: mean";
        String ancillary_variables "dopy";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    dopx {
        Int16 _FillValue -32767;
        String long_name "longitudinal dilution of precision";
        String comment "The longitudinal dilution of precision (dopx) represents the contribution of radial geometry to uncertainty in the eastward velocity estimate (u).";
        Float32 scale_factor 0.01;
        String grid_mapping "wgs84";
        String coordinates "depth";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    dopy {
        Int16 _FillValue -32767;
        String long_name "latitudinal dilution of precision";
        String comment "The latitudinal dilution of precision (dopy) represents the contribution of radial geometry to uncertainty in the northward velocity estimate (v).";
        Float32 scale_factor 0.01;
        String grid_mapping "wgs84";
        String coordinates "depth";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    hdop {
        Int16 _FillValue -32767;
        String long_name "horizontal dilution of precision";
        String comment "The horizontal dilution of precision (hdop) is the vector length (magnitude) of the eastward (dopx) and northward (dopy) dilution of precision.  It represents the contribution of radial geometry to the overall uncertainty in the total velocity (u and v) estimate.";
        Float32 scale_factor 0.01;
        String grid_mapping "wgs84";
        String coordinates "depth";
        String ancillary_variables "dopx dopy";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    number_of_sites {
        String _Unsigned "false";
        Int16 _FillValue -127;
        String long_name "number of contributing radars";
        String units "count";
        String comment "Number of radars contributing radials to the total solution";
        String grid_mapping "wgs84";
        String coordinates "depth";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    number_of_radials {
        Int16 _FillValue -32767;
        String long_name "number of contributing radials";
        String units "count";
        String comment "Number of radials contributing to the total solution";
        String grid_mapping "wgs84";
        String coordinates "depth";
        UInt32 _ChunkSizes 1, 1099, 700;
    }
    NC_GLOBAL {
        String Conventions "CF-1.12,ACDD-1.3";
        String id_HFR "202610141600iooshfruwlsRTVuswc2km";
        String id "8a1a90e7-958c-40fd-84b5-174d5b235167";
        String date_created "2026-10-14T17:39:55Z";
        String source "RDLm_ACI1_2026_10_14_1600.ruv, RDLm_ARG1_2026_10_14_1600.ruv, RDLm_BML1_2026_10_14_1600.ruv, RDLm_COP1_2026_10_14_1600.ruv, RDLm_CRIS_2026_10_14_1600.ruv, RDLm_ESTR_2026_10_14_1600.ruv, RDLm_EXPL_2026_10_14_1600.ruv, RDLm_FBK1_2026_10_14_1600.ruv, RDLm_FORT_2026_10_14_1600.ruv, RDLm_GCVE_2026_10_14_1600.ruv, RDLm_GCYN_2026_10_14_1600.ruv, RDLm_LUIS_2026_10_14_1600.ruv, RDLm_MBAC_2026_10_14_1600.ruv, RDLm_MGS1_2026_10_14_1600.ruv, RDLm_MLML_2026_10_14_1600.ruv, RDLm_MONT_2026_10_14_1600.ruv, RDLm_NIC1_2026_10_14_1600.ruv, RDLi_PBON_2026_10_14_1600.ruv, RDLm_PPNS_2026_10_14_1600.ruv, RDLm_PREY_2026_10_14_1600.ruv, RDLm_PTC1_2026_10_14_1600.ruv, RDLm_PTM1_2026_10_14_1600.ruv, RDLm_RFG1_2026_10_14_1600.ruv, RDLm_RTC1_2026_10_14_1600.ruv, RDLm_SAUS_2026_10_14_1600.ruv, RDLm_SCCI_2026_10_14_1600.ruv, RDLi_SCDB_2026_10_14_1600.ruv, RDLm_SCDH_2026_10_14_1600.ruv, RDLm_SCI1_2026_10_14_1600.ruv, RDLi_SCPF_2026_10_14_1600.ruv, RDLm_SCRZ_2026_10_14_1600.ruv, RDLi_SCTB_2026_10_14_1600.ruv, RDLm_SDBP_2026_10_14_1600.ruv, RDLi_SDCP_2026_10_14_1600.ruv, RDLm_SDDP_2026_10_14_1600.ruv, RDLm_SDSE_2026_10_14_1600.ruv, RDLm_SDUT_2026_10_14_1600.ruv, RDLm_SLID_2026_10_14_1600.ruv, RDLm_SNI1_2026_10_14_1600.ruv, RDLm_SSD1_2026_10_14_1600.ruv, RDLi_WSH1_2026_10_14_1600.ruv, RDLi_YHS2_2026_10_14_1600.ruv";
        String program "Integrated Ocean Observing System (IOOS)";
        String project "NESDIS Common Cloud Framework.";
        String title "Near-Real Time Surface Ocean Velocity, U.S. West Coast, 2km Resolution";
        String summary "Surface ocean velocities estimated from HF-Radar are representative of the upper 1.0 meters of the ocean.  The main objective of near-real time processing is to produce the best product from available data at the time of processing. Radial velocity measurements are obtained from individual radar sites through the U.S. HF-Radar Network. Hourly radial data are processed by unweighted least squares on a 2km resolution grid of the U.S. West Coast to produce near real-time surface current maps.";
        String instrument "OCEAN SURFACE CURRENT RADAR";
        String keywords "NOAA, IOOS, CODAR SeaSonde, Earth Science, Oceans, Coastal Processes, Marine Environment Monitoring, Ocean Circulation, Ocean Currents, Surface Currents, Wind-Driven Circulation, Tides, Tidal Currents, Pacific Ocean, North Pacific Ocean";
        Float32 geospatial_lat_min 30.25;
        Float32 geospatial_lat_max 49.99204;
        String geospatial_lat_resolution "2 km";
        String geospatial_lat_units "degrees_north";
        Float32 geospatial_lon_min -130.36;
        Float32 geospatial_lon_max -115.805565;
        String geospatial_lon_resolution "2 km";
        String geospatial_lon_units "degrees_east";
        String processing_level "NOAA Level 3. Near real-time dataset with automated data acquisition and processing quality control.";
        String history "HFRNet v1r0";
        String references "Terrill, E. et al., 2006. Data Management and Real-time Distribution in the HF-Radar National Network. Proceedings of the MTS/IEEE Oceans 2006 Conference, Boston MA, September 2006.";
        String institution "DOC/NOAA/NESDIS/OSPO > Office of Satellite and Product Operations, NESDIS, NOAA, U.S. Department of Commerce.";
        String creator_type "institution";
        String creator_name "NOAA IOOS Surface Currents Program";
        String creator_email "data.ioos@noaa.gov";
        String creator_url "https://ioos.noaa.gov/project/hf-radar/";
        String naming_authority "gov.noaa.nesdis.ncei";
        String standard_name_vocabulary "CF Standard Name Table, Version 89";
        String keywords_vocabulary "Global Change Master Directory (GCMD) Keywords, Version 21.2";
        String instrument_vocabulary "Global Change Master Directory (GCMD) Keywords, Version 21.2";
        String format_version "2.0.00";
        String product_version "v1r0";
        String time_coverage_start "2026-10-14T16:00:00Z";
        String time_coverage_end "2026-10-14T16:00:00Z";
        String cdm_data_type "Grid";
        String day_night_data_flag "both";
        String metadata_link "https://cdn.ioos.noaa.gov/media/2017/12/hfradar_nodc_sep_2014.pdf";
        String platform "CODAR SeaSonde, COASTAL STATIONS";
        String platform_vocabulary "Global Change Master Directory (GCMD) Keywords, Version 21.2";
        String production_environment "INT";
        String production_site "PLAIT";
        String publisher_email "espcoperations@noaa.gov";
        String publisher_type "institution";
        String publisher_name "DOC/NOAA/NESDIS/OSPO > Office of Satellite and Product Operations, NESDIS, NOAA, U.S. Department of Commerce.";
        String publisher_url "http://www.ospo.noaa.gov";
    }
}
