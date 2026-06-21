#!/usr/bin/env bash
# ============================================================================
# install_wrf.sh — Install + compile WRF (em_real) and WPS on this VM.
#
# Target: Debian 12 (bookworm). Verified on aarch64; works on x86_64 too.
# Strategy: use distro packages for the toolchain + NetCDF/HDF5/MPI/Jasper,
# consolidate NetCDF into a single prefix (WRF wants one $NETCDF; Debian
# multiarch splits C and Fortran), then build WRF then WPS from source.
#
# Idempotent: re-running skips downloads/compiles that already succeeded.
# Honest: every stage fails loudly; nothing is faked.
#
# Env (override before running):
#   WRF_ROOT   install root              (default: /opt/wrf)
#   WRF_VER    WRF version               (default: 4.6.1)
#   WPS_VER    WPS version               (default: 4.6.0)
#   JOBS       parallel compile jobs     (default: nproc)
# ============================================================================
set -Eeuo pipefail

WRF_ROOT="${WRF_ROOT:-/opt/wrf}"
WRF_VER="${WRF_VER:-4.6.1}"
WPS_VER="${WPS_VER:-4.6.0}"
JOBS="${JOBS:-$(nproc)}"

DEPS="$WRF_ROOT/deps"
SRC="$WRF_ROOT/src"
NCDIR="$DEPS/netcdf"                 # consolidated NetCDF prefix
LOG="$WRF_ROOT/install.log"

log()  { echo -e "\033[36m[install]\033[0m $*" | tee -a "$LOG"; }
err()  { echo -e "\033[31m[install:ERROR]\033[0m $*" | tee -a "$LOG" >&2; }
die()  { err "$*"; exit 1; }
trap 'err "failed at line $LINENO (see $LOG)"' ERR

sudo mkdir -p "$WRF_ROOT" "$DEPS" "$SRC"
sudo chown -R "$(id -u):$(id -g)" "$WRF_ROOT"
: > "$LOG"
log "WRF_ROOT=$WRF_ROOT  WRF_VER=$WRF_VER  WPS_VER=$WPS_VER  JOBS=$JOBS  arch=$(uname -m)"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "1/6 apt-get update + install build deps"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y >>"$LOG" 2>&1
sudo apt-get install -y --no-install-recommends \
  gcc g++ gfortran make cmake m4 perl tcsh \
  autotools-dev \
  python3 python3-pip python3-numpy python3-netcdf4 \
  wget curl git ca-certificates file \
  zlib1g-dev libpng-dev \
  libnetcdf-dev libnetcdff-dev netcdf-bin \
  libhdf5-dev \
  mpich libmpich-dev \
  >>"$LOG" 2>&1 || die "apt install failed — inspect $LOG"
# NOTE: jasper (GRIB2 for WPS/ungrib) is NOT packaged in Debian 12 (removed over CVEs),
# so it is built from source below rather than installed via apt.

# Resolve multiarch triplet (e.g. aarch64-linux-gnu / x86_64-linux-gnu)
TRIPLET="$(gcc -dumpmachine)"
LIBDIR="/usr/lib/$TRIPLET"
log "multiarch triplet=$TRIPLET  libdir=$LIBDIR"

# ---------------------------------------------------------------------------
# 2. Consolidate NetCDF into a single prefix for WRF/WPS
# ---------------------------------------------------------------------------
log "2/6 consolidating NetCDF into $NCDIR"
rm -rf "$NCDIR"; mkdir -p "$NCDIR/include" "$NCDIR/lib" "$NCDIR/bin"
# headers + fortran modules
for f in /usr/include/netcdf.h /usr/include/netcdf*.h /usr/include/netcdf.inc \
         /usr/include/netcdf.mod /usr/include/netcdf*.mod /usr/include/typesizes.mod; do
  [ -e "$f" ] && ln -sf "$f" "$NCDIR/include/" || true
done
# libs (C + Fortran)
for f in "$LIBDIR"/libnetcdf.so* "$LIBDIR"/libnetcdf.a \
         "$LIBDIR"/libnetcdff.so* "$LIBDIR"/libnetcdff.a; do
  [ -e "$f" ] && ln -sf "$f" "$NCDIR/lib/" || true
done
# nc-config / nf-config convenience
for b in nc-config nf-config ncdump ncgen; do
  p="$(command -v $b || true)"; [ -n "$p" ] && ln -sf "$p" "$NCDIR/bin/" || true
done
[ -e "$NCDIR/include/netcdf.h" ]  || die "netcdf.h not found after consolidation"
[ -e "$NCDIR/include/netcdf.mod" ] || die "netcdf.mod (Fortran) not found — libnetcdff-dev missing?"
ls "$NCDIR"/lib/libnetcdff.* >/dev/null 2>&1 || die "libnetcdff not found — libnetcdff-dev missing?"

# Consolidate HDF5 into a prefix with the exact lib names WRF expects.
# Debian's NetCDF is built against HDF5, so WRF links -lhdf5{,_fortran,_hl,_hl_fortran}
# from $HDF5/lib — but Debian ships them suffixed "_serial" / under hdf5/serial/ with
# one quirky name (libhdf5hl_fortran, missing an underscore). Bridge the names here.
HDF5DIR="$DEPS/hdf5"
log "2b/6 consolidating HDF5 into $HDF5DIR"
rm -rf "$HDF5DIR"; mkdir -p "$HDF5DIR/include" "$HDF5DIR/lib"
H5SRC=""
for d in "$LIBDIR/hdf5/serial" "$LIBDIR"; do
  [ -e "$d/libhdf5.so" ] || [ -e "$d/libhdf5_serial.so" ] && { H5SRC="$d"; break; }
done
[ -n "$H5SRC" ] || die "HDF5 libs not found under $LIBDIR — libhdf5-dev missing?"
link_h5() { # link_h5 <wanted-basename> <candidate...>
  local want="$1"; shift; local c
  for c in "$@"; do
    for ext in so a; do
      if [ -e "$H5SRC/$c.$ext" ]; then ln -sf "$H5SRC/$c.$ext" "$HDF5DIR/lib/$want.$ext"; fi
    done
  done
}
link_h5 libhdf5            libhdf5            libhdf5_serial
link_h5 libhdf5_fortran    libhdf5_fortran    libhdf5_serial_fortran
link_h5 libhdf5_hl         libhdf5_hl         libhdf5_serial_hl
link_h5 libhdf5_hl_fortran libhdf5_hl_fortran libhdf5hl_fortran libhdf5_serialhl_fortran libhdf5_serial_hl_fortran
for h in "$LIBDIR/hdf5/serial/include"/*.h /usr/include/hdf5/serial/*.h /usr/include/hdf5.h; do
  [ -e "$h" ] && ln -sf "$h" "$HDF5DIR/include/" || true
done
for w in libhdf5 libhdf5_fortran libhdf5_hl libhdf5_hl_fortran; do
  ls "$HDF5DIR/lib/$w".* >/dev/null 2>&1 || die "HDF5 consolidation: missing $w (looked in $H5SRC)"
done

# Jasper (for WPS GRIB2 / ungrib). Debian 12 has no jasper package, so prefer a
# distro copy if present, otherwise build the WPS-compatible jasper-1.900.1.
if [ -e "/usr/include/jasper/jasper.h" ] && ls "$LIBDIR"/libjasper.* >/dev/null 2>&1; then
  JASPERINC="/usr/include"
  JASPERLIB="$LIBDIR"
  log "jasper: using distro copy ($JASPERLIB)"
else
  JPREFIX="$DEPS/jasper"
  if [ ! -e "$JPREFIX/lib/libjasper.a" ] && [ ! -e "$JPREFIX/lib/libjasper.so" ]; then
    log "jasper: building jasper-1.900.1 from source (not packaged in Debian 12)"
    cd "$SRC"
    if [ ! -f jasper-1.900.1.tar.gz ]; then
      wget -q -O jasper-1.900.1.tar.gz \
        "https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/jasper-1.900.1.tar.gz" \
        || die "jasper source download failed"
    fi
    rm -rf jasper-1.900.1; tar -xzf jasper-1.900.1.tar.gz
    cd jasper-1.900.1
    # jasper-1.900.1 ships a 2003 config.guess that doesn't know aarch64 — refresh it.
    for cf in config.guess config.sub; do
      [ -e "/usr/share/misc/$cf" ] && cp -f "/usr/share/misc/$cf" "acaux/$cf"
    done
    chmod +x acaux/config.* 2>/dev/null || true
    # gcc>=10 needs -fcommon (legacy multiple-definition); silence legacy warnings
    ./configure --prefix="$JPREFIX" --build="$TRIPLET" \
      CFLAGS="-O2 -fPIC -fcommon -Wno-implicit-function-declaration -Wno-error" \
      >>"$LOG" 2>&1 || die "jasper configure failed — see $LOG"
    # jasper is tiny; build serially — its Makefile has a parallel race on libtool deps.
    make >>"$LOG" 2>&1 || die "jasper build failed — see $LOG"
    make install >>"$LOG" 2>&1 || die "jasper install failed — see $LOG"
  fi
  JASPERINC="$JPREFIX/include"
  JASPERLIB="$JPREFIX/lib"
  log "jasper: built at $JPREFIX"
fi
[ -e "$JASPERINC/jasper/jasper.h" ] || die "jasper.h not found at $JASPERINC after install"
ls "$JASPERLIB"/libjasper.* >/dev/null 2>&1 || die "libjasper not found at $JASPERLIB after install"

# ---------------------------------------------------------------------------
# 3. Build environment exported to compilers
# ---------------------------------------------------------------------------
export NETCDF="$NCDIR"
export NETCDF_classic=1
export HDF5="$HDF5DIR"
export JASPERLIB JASPERINC
export PATH="$NCDIR/bin:$PATH"
export LD_LIBRARY_PATH="$NCDIR/lib:$HDF5DIR/lib:$JASPERLIB:$LIBDIR:${LD_LIBRARY_PATH:-}"
# gfortran >=10 needs this to compile WRF's legacy argument mismatches
export FFLAGS="-fallow-argument-mismatch -fallow-invalid-boz ${FFLAGS:-}"
export FCFLAGS="$FFLAGS"
log "3/6 env: NETCDF=$NETCDF HDF5=$HDF5 JASPERLIB=$JASPERLIB"

# ---------------------------------------------------------------------------
# 4. Download + compile WRF (em_real)
# ---------------------------------------------------------------------------
WRF_TGZ="$SRC/wrf-$WRF_VER.tar.gz"
WRF_DIR=""
log "4/6 WRF $WRF_VER"
if [ ! -f "$WRF_TGZ" ]; then
  wget -q -O "$WRF_TGZ" \
    "https://github.com/wrf-model/WRF/releases/download/v$WRF_VER/v$WRF_VER.tar.gz" \
    || die "WRF download failed"
fi
tar -xzf "$WRF_TGZ" -C "$SRC"
WRF_DIR="$(find "$SRC" -maxdepth 1 -type d -iname 'WRF*' | head -1)"
[ -n "$WRF_DIR" ] || die "extracted WRF dir not found"
ln -sfn "$WRF_DIR" "$WRF_ROOT/WRF"
log "WRF dir: $WRF_DIR"

cd "$WRF_DIR"
if [ ! -f main/wrf.exe ]; then
  # Pick the GNU (gfortran/gcc) dmpar option number from the configure menu.
  MENU="$(printf '999\n1\n' | ./configure 2>&1 || true)"
  echo "$MENU" >>"$LOG"
  DMPAR="$(echo "$MENU" | grep -iE 'GNU .*gfortran' | grep -oE '[0-9]+\. *\(dmpar\)' \
            | grep -oE '^[0-9]+' | head -1)"
  if [ -z "$DMPAR" ]; then
    # fallback: 3rd number on the GNU line (serial smpar dmpar dm+sm)
    DMPAR="$(echo "$MENU" | grep -iE 'GNU .*gfortran' | grep -oE '[0-9]+' | sed -n '3p')"
  fi
  [ -n "$DMPAR" ] || die "could not find GNU dmpar option in configure menu (ARM unsupported?) — see $LOG"
  log "WRF configure: GNU dmpar option = $DMPAR (nesting=1)"
  printf '%s\n1\n' "$DMPAR" | ./configure >>"$LOG" 2>&1 || die "WRF configure failed"
  # Force-enable the gfortran-10+ flag in the generated config in case it was dropped
  if ! grep -q 'fallow-argument-mismatch' configure.wrf; then
    sed -i 's/^\(FCFLAGS *=.*\)/\1 -fallow-argument-mismatch -fallow-invalid-boz/' configure.wrf || true
    sed -i 's/^\(FFLAGS *=.*\)/\1 -fallow-argument-mismatch -fallow-invalid-boz/' configure.wrf || true
  fi
  log "compiling WRF em_real (this takes a while)…"
  ./compile -j "$JOBS" em_real >>"$LOG" 2>&1 || true   # compile reports success via binaries
fi
[ -f main/wrf.exe ]  || die "WRF compile produced no main/wrf.exe — see $LOG"
[ -f main/real.exe ] || die "WRF compile produced no main/real.exe — see $LOG"
log "WRF OK: $(ls -la main/wrf.exe main/real.exe | awk '{print $NF}' | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 5. Download + compile WPS
# ---------------------------------------------------------------------------
export WRF_DIR
WPS_TGZ="$SRC/wps-$WPS_VER.tar.gz"
log "5/6 WPS $WPS_VER"
if [ ! -f "$WPS_TGZ" ]; then
  wget -q -O "$WPS_TGZ" \
    "https://github.com/wrf-model/WPS/archive/refs/tags/v$WPS_VER.tar.gz" \
    || die "WPS download failed"
fi
tar -xzf "$WPS_TGZ" -C "$SRC"
WPS_DIR="$(find "$SRC" -maxdepth 1 -type d -iname 'WPS*' | head -1)"
[ -n "$WPS_DIR" ] || die "extracted WPS dir not found"
ln -sfn "$WPS_DIR" "$WRF_ROOT/WPS"
log "WPS dir: $WPS_DIR"

cd "$WPS_DIR"
if [ ! -f geogrid.exe ] || [ ! -f ungrib.exe ] || [ ! -f metgrid.exe ]; then
  # WPS 4.x ships no aarch64 platform stanza, so on ARM the configure menu lists ZERO
  # options and then infinite-loops on EOF ("Enter selection [1-0]"). Add aarch64 to the
  # generic gfortran/gcc x86_64 stanza (its flags are arch-neutral; WRF builds with the
  # same compiler) so the dmpar/GRIB2 options appear. Idempotent.
  if ! grep -q 'Linux x86_64 aarch64, gfortran' arch/configure.defaults; then
    sed -i 's/^#ARCH    Linux x86_64, gfortran /#ARCH    Linux x86_64 aarch64, gfortran /' \
      arch/configure.defaults
  fi
  # Capture the menu WITHOUT triggering the EOF spin: feed /dev/null and cap with head so
  # SIGPIPE kills configure right after it prints the option list.
  MENU="$(./configure </dev/null 2>&1 | head -60 || true)"
  echo "$MENU" >>"$LOG"
  # WPS GRIB2-enabled options have NO 'grib2' in their label; the GRIB2-DISABLED
  # ones are labelled '..._NO_GRIB2'. So pick a gfortran dmpar option that does
  # NOT contain NO_GRIB2. Fall back to serial (GRIB2), then any gfortran option.
  WOPT="$(echo "$MENU" | grep -iE 'gfortran' | grep -iE 'dmpar' | grep -ivE 'no_grib2' \
           | grep -oE '^[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1)"
  [ -z "$WOPT" ] && WOPT="$(echo "$MENU" | grep -iE 'gfortran' | grep -iE 'serial' | grep -ivE 'no_grib2' \
           | grep -oE '^[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1)"
  [ -z "$WOPT" ] && WOPT="$(echo "$MENU" | grep -iE 'gfortran' | grep -oE '^[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1)"
  [ -n "$WOPT" ] || die "could not find a gfortran WPS configure option — see $LOG"
  log "WPS configure option = $WOPT"
  # timeout-guard the real configure too: a valid option won't loop, but never hang the build.
  printf '%s\n' "$WOPT" | timeout 120 ./configure >>"$LOG" 2>&1 || true
  [ -f configure.wps ] || die "WPS configure produced no configure.wps — see $LOG"
  # Ensure WRF_DIR + gfortran flags land in configure.wps
  sed -i "s#^WRF_DIR *=.*#WRF_DIR = $WRF_DIR#" configure.wps || true
  if ! grep -q 'fallow-argument-mismatch' configure.wps; then
    sed -i 's/^\(FFLAGS *=.*\)/\1 -fallow-argument-mismatch/' configure.wps || true
  fi
  log "compiling WPS…"
  ./compile >>"$LOG" 2>&1 || true
fi
for b in geogrid.exe ungrib.exe metgrid.exe; do
  [ -e "$b" ] || die "WPS compile produced no $b — see $LOG"
done
log "WPS OK: geogrid.exe ungrib.exe metgrid.exe present"

# ---------------------------------------------------------------------------
# 6. Persist environment for check/run scripts
# ---------------------------------------------------------------------------
cat > "$WRF_ROOT/wrf_env.sh" <<EOF
# sourced by check_wrf_install.sh and run_test_case.sh
export WRF_ROOT="$WRF_ROOT"
export WRF_DIR="$WRF_DIR"
export WPS_DIR="$WPS_DIR"
export NETCDF="$NCDIR"
export HDF5="$HDF5DIR"
export JASPERLIB="$JASPERLIB"
export JASPERINC="$JASPERINC"
export PATH="$NCDIR/bin:\$PATH"
export LD_LIBRARY_PATH="$NCDIR/lib:$HDF5DIR/lib:$JASPERLIB:$LIBDIR:\${LD_LIBRARY_PATH:-}"
EOF
log "6/6 wrote $WRF_ROOT/wrf_env.sh"
log "DONE. WRF=$WRF_DIR  WPS=$WPS_DIR"
echo
echo "Next:  bash infra/check_wrf_install.sh"
