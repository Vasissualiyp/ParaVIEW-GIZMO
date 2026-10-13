{
  description = "ParaView + Meshoid dev shell for a GIZMO (meshless) data plugin";

  inputs = {
    # Pinned to a rev whose paraview/vtk are in cache.nixos.org (no local VTK build).
    nixpkgs.url = "github:nixos/nixpkgs/151fa4e8ddfdd8dd25d945ad94ed54a13de9f6e4";
    flake-utils.url = "github:numtide/flake-utils";
    # Same commit as the dev shell's editable ./meshoid clone; used for the server image.
    meshoid-src = {
      url = "github:mikegrudic/meshoid/ad329662ae91e7f39ca113943085165f8f18eaf8";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, meshoid-src }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };

        # ParaView's embedded interpreter is VTK's python. Everything that
        # pvpython / the GUI imports must be built against this exact python.
        py = pkgs.vtk-full.vtkPackages.python3Packages;

        # meshoid itself is used from the local clone ./meshoid (editable-style,
        # via PYTHONPATH in shellHook); nix only supplies its dependencies.
        pythonEnv = py.python.withPackages (ps: with ps; [
          numpy scipy numba h5py astropy matplotlib
          ipython pytest
        ]);
        # Runtime environment shared by the dev shell and the server image.
        runtimeEnv = {
          PV_GIZMO_PYTHON = "${pythonEnv}/bin/python3";
        };

        # The plugin + core package as installed in the image (/opt/pvgizmo).
        pvgizmoSrc = pkgs.runCommand "pvgizmo-src" { } ''
          mkdir -p $out/opt/pvgizmo $out/opt/meshoid
          cp -r ${./pvgizmo} $out/opt/pvgizmo/pvgizmo
          cp -r ${./plugin} $out/opt/pvgizmo/plugin
          cp -r ${meshoid-src}/src $out/opt/meshoid/src
        '';

        # Headless pvserver for clusters (Trillium): ParaView 6.0.1 identical to the
        # dev shell's client (client and server versions must match), meshoid, the
        # plugins, and Mesa (EGL + llvmpipe) for offscreen rendering without GPU/X.
        #   nix build .#pvserver-image   -> docker-archive tarball
        #   apptainer build pvgizmo.sif docker-archive://result
        pvserverImage = pkgs.dockerTools.buildLayeredImage {
          name = "pvgizmo-pvserver";
          tag = "6.0.1";
          contents = [ pkgs.paraview pythonEnv pvgizmoSrc pkgs.mesa pkgs.bashInteractive pkgs.coreutils
                       pkgs.cacert ];
          extraCommands = "mkdir -p tmp && chmod 1777 tmp";
          config = {
            Entrypoint = [ "${pkgs.paraview}/bin/pvserver" ];
            Cmd = [ "--force-offscreen-rendering" ];
            Env = [
              "PV_GIZMO_PYTHON=${pythonEnv}/bin/python3"
              "PV_GIZMO_ROOT=/opt/pvgizmo"
              "PYTHONPATH=/opt/pvgizmo:/opt/meshoid/src:${pythonEnv}/${pythonEnv.sitePackages}"
              "PV_PLUGIN_PATH=/opt/pvgizmo/plugin"
              "NUMBA_CACHE_DIR=/tmp/numba-cache"
              "__EGL_VENDOR_LIBRARY_FILENAMES=${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json"
              "LIBGL_DRIVERS_PATH=${pkgs.mesa}/lib/dri"
              "LD_LIBRARY_PATH=${pkgs.mesa}/lib"
              "PATH=${pkgs.paraview}/bin:${pythonEnv}/bin:${pkgs.coreutils}/bin:${pkgs.bashInteractive}/bin"
            ];
          };
        };
      in
      {
        packages.pvserver-image = pvserverImage;

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            paraview
            pythonEnv
            # C++ plugin toolchain (ParaView's CMake config propagates qt6/vtk)
            cmake ninja gcc gdb pkg-config
            qt6Packages.wrapQtAppsHook
            hdf5
            git
            xvfb-run   # headless GUI tests: xvfb-run -a paraview --script=... --exit
            xdotool    # real mouse drags in the headless GUI interactivity test
            apptainer  # run/test the cluster pvserver image (packages.pvserver-image)
          ];

          shellHook = ''
            export PV_GIZMO_ROOT=$PWD
            # meshoid is a git-ignored local clone (editable). Recreate it if missing.
            if [ ! -d "$PWD/meshoid" ]; then
              git clone -q https://github.com/mikegrudic/meshoid "$PWD/meshoid" \
                && git -C "$PWD/meshoid" checkout -q ad32966
            fi
            # sys.executable is pvpython/paraview, which can't run multiprocessing
            # helpers (numba's parallel init spawns resource_tracker). Plugins call
            # multiprocessing.set_executable($PV_GIZMO_PYTHON) before importing numba.
            export PV_GIZMO_PYTHON=${runtimeEnv.PV_GIZMO_PYTHON}
            # Make pvgizmo, meshoid and pythonEnv's site-packages (numba, h5py...) visible
            # inside paraview / pvpython, which use VTK's bare interpreter.
            export PYTHONPATH=$PWD:$PWD/meshoid/src:${pythonEnv}/${pythonEnv.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}
            # Auto-load our Python plugin(s) when present.
            if [ -d "$PWD/plugin" ]; then
              export PV_PLUGIN_PATH=$PWD/plugin''${PV_PLUGIN_PATH:+:$PV_PLUGIN_PATH}
            fi
            # Some desktop sessions (e.g. Hyprland setups) export QT_XCB_GL_INTEGRATION=none.
            # ParaView runs on xcb and its main window needs GL, so it then never gets
            # mapped: only the Welcome dialog appears and closing it leaves nothing.
            unset QT_XCB_GL_INTEGRATION
            export NUMBA_CACHE_DIR=''${XDG_CACHE_HOME:-$HOME/.cache}/numba
            echo "ParaView $(pvpython --version 2>/dev/null | tail -1) | meshoid dev shell"
          '';
        };
      });
}
