------- OpenFAST INPUT FILE -------------------------------------------
Generated with AeroElasticSE FAST driver
---------------------- SIMULATION CONTROL --------------------------------------
False                  Echo        - Echo input data to <RootName>.ech (flag)
"FATAL"                AbortLevel  - Error level when simulation should abort (string) {"WARNING", "SEVERE", "FATAL"}
          1   TMax            - Total run time (s)
      0.00125   DT              - Recommended module time step (s)
2                      InterpOrder - Interpolation order for input/output time history (-) {1=linear, 2=quadratic}
1                      NumCrctn    - Number of correction iterations (-) {0=explicit calculation, i.e., no corrections}
99999.0                DT_UJac     - Time between calls to get Jacobians (s)
1000000.0              UJacSclFact - Scaling factor used in Jacobians (-)
---------------------- FEATURE SWITCHES AND FLAGS ------------------------------
          2   CompElast       - Compute structural dynamics (switch) {1=ElastoDyn; 2=ElastoDyn + BeamDyn for blades}
          1   CompInflow      - Compute inflow wind velocities (switch) {0=still air; 1=InflowWind; 2=external from OpenFOAM}
          3   CompAero        - Compute aerodynamic loads (switch) {0=None; 1=AeroDyn v14; 2=AeroDyn v15}
          0   CompServo       - Compute control and electrical-drive dynamics (switch) {0=None; 1=ServoDyn}
          0   CompSeaSt   - Compute sea state information (switch) {0=None; 1=SeaState}
          0   CompHydro       - Compute hydrodynamic loads (switch) {0=None; 1=HydroDyn}
          0   CompSub         - Compute sub-structural dynamics (switch) {0=None; 1=SubDyn; 2=External Platform MCKF}
          0   CompMooring     - Compute mooring system (switch) {0=None; 1=MAP++; 2=FEAMooring; 3=MoorDyn; 4=OrcaFlex}
          0   CompIce         - Compute ice loads (switch) {0=None; 1=IceFloe; 2=IceDyn}
          0   MHK             - MHK turbine type (switch) {0=Not an MHK turbine; 1=Fixed MHK turbine; 2=Floating MHK turbine}
---------------------- ENVIRONMENTAL CONDITIONS --------------------------------
          0   Gravity         - Gravitational acceleration (m/s^2)
1.225                  AirDens     - Air density (kg/m^3)
1025                   WtrDens     - Water density (kg/m^3)
1.464e-05              KinVisc     - Kinematic viscosity of working fluid (m^2/s)
335                    SpdSound    - Speed of sound in working fluid (m/s)
103500                 Patm        - Atmospheric pressure (Pa) [used only for an MHK turbine cavitation check]
1700                   Pvap        - Vapour pressure of working fluid (Pa) [used only for an MHK turbine cavitation check]
300                    WtrDpth     - Water depth (m)
0                      MSL2SWL     - Offset between still-water level and mean sea level (m) [positive upward]
---------------------- INPUT FILES ---------------------------------------------
"00_IEA-10.0-198-RWT_ElastoDyn.dat"   EDFile          - Name of file containing ElastoDyn input parameters (quoted string)
"00_IEA-10.0-198-RWT_BeamDyn.dat"   BDBldFile(1)    - Name of file containing BeamDyn input parameters for blade 1 (quoted string)
"00_IEA-10.0-198-RWT_BeamDyn.dat"   BDBldFile(2)    - Name of file containing BeamDyn input parameters for blade 2 (quoted string)
"00_IEA-10.0-198-RWT_BeamDyn.dat"   BDBldFile(3)    - Name of file containing BeamDyn input parameters for blade 3 (quoted string)
"00_IEA-10.0-198-RWT_InflowFile.dat"   InflowFile      - Name of file containing inflow wind input parameters (quoted string)
"00_IEA-10.0-198-RWT_AeroDyn15.dat"   AeroFile        - Name of file containing aerodynamic input parameters (quoted string)
"00_IEA-10.0-198-RWT_ServoDyn.dat"   ServoFile       - Name of file containing control and electrical-drive input parameters (quoted string)
"unused"               SeaStFile   - Name of file containing sea state input parameters (quoted string)
"00_IEA-10.0-198-RWT_HydroDyn.dat"   HydroFile       - Name of file containing hydrodynamic input parameters (quoted string)
"00_IEA-10.0-198-RWT_SubDyn.dat"   SubFile         - Name of file containing sub-structural input parameters (quoted string)
"FileNotFound"   MooringFile     - Name of file containing mooring system input parameters (quoted string)
"FileNotFound"   IceFile         - Name of file containing ice input parameters (quoted string)
---------------------- OUTPUT --------------------------------------------------
False                  SumPrint    - Print summary data to "<RootName>.sum" (flag)
          1   SttsTime        - Amount of time between screen status messages (s)
99999.0                ChkptTime   - Amount of time between creating checkpoint files for potential restart (s)
default                DT_Out      - Time step for tabular output (s) (or "default")
0.0                    TStart      - Time to begin tabular output (s)
          1   OutFileFmt      - Format for tabular (time-marching) output file (switch) {1: text file [<RootName>.out], 2: binary file [<RootName>.outb], 3: both}
True                   TabDelim    - Use tab delimiters in text tabular output file? (flag) {uses spaces if false}
"ES20.12E3"   OutFmt          - Format used for text tabular output, excluding the time channel.  Resulting field should be 10 characters. (quoted string)
---------------------- LINEARIZATION -------------------------------------------
       true   Linearize       - Linearization analysis (flag)
      false   CalcSteady      - Calculate a steady-state periodic operating point before linearization? [unused if Linearize=False] (flag)
          3   TrimCase        - Controller parameter to be trimmed {1:yaw; 2:torque; 3:pitch} [used only if CalcSteady=True] (-)
      0.001   TrimTol         - Tolerance for the rotational speed convergence [used only if CalcSteady=True] (-)
       0.01   TrimGain        - Proportional gain for the rotational speed error (>0) [used only if CalcSteady=True] (rad/(rad/s) for yaw or pitch; Nm/(rad/s) for torque)
          0   Twr_Kdmp        - Damping factor for the tower [used only if CalcSteady=True] (N/(m/s))
          0   Bld_Kdmp        - Damping factor for the blades [used only if CalcSteady=True] (N/(m/s))
          1   NLinTimes       - Number of times to linearize (-) [>=1] [unused if Linearize=False]
          0   LinTimes        - List of times at which to linearize (s) [1 to NLinTimes] [used only when Linearize=True and CalcSteady=False]
          2   LinInputs       - Inputs included in linearization (switch) {0=none; 1=standard; 2=all module inputs (debug)} [unused if Linearize=False]
          2   LinOutputs      - Outputs included in linearization (switch) {0=none; 1=from OutList(s); 2=all module outputs (debug)} [unused if Linearize=False]
       true   LinOutJac       - Include full Jacobians in linearization output (for debug) (flag) [unused if Linearize=False; used only if LinInputs=LinOutputs=2]
       true   LinOutMod       - Write module-level linearization output files in addition to output for full system? (flag) [unused if Linearize=False]
---------------------- VISUALIZATION ------------------------------------------
          3   WrVTK           - VTK visualization data output: (switch) {0=none; 1=initialization data only; 2=animation}
          2   VTK_type        - Type of VTK visualization data: (switch) {1=surfaces; 2=basic meshes (lines/points); 3=all meshes (debug)} [unused if WrVTK=0]
False                  VTK_fields  - Write mesh fields to VTK data files? (flag) {true/false} [unused if WrVTK=0]
         24   VTK_fps         - Frame rate for VTK output (frames per second){will use closest integer multiple of DT} [used only if WrVTK=2]
