{
  description = "Sana dev shell — Node for app/, plus the shared-library stack the Expo web dev server needs on Linux to launch React Native DevTools (a bundled Electron/Chromium binary).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        electronRuntimeLibs = with pkgs; [
          glib
          nspr
          nss
          dbus
          atk
          at-spi2-atk
          at-spi2-core
          cups
          cairo
          gtk3
          pango
          libx11
          libxcomposite
          libxdamage
          libxext
          libxfixes
          libxrandr
          libxcb
          libxkbcommon
          alsa-lib
          expat
          libgbm
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.nodejs_22 ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux electronRuntimeLibs;

          shellHook = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath electronRuntimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '';
        };
      });
}
