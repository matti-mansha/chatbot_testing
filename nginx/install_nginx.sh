#!/usr/bin/env bash
# install_nginx.sh — one-command nginx reverse proxy installer for the dashboard.
#
# What this does (sudo required):
#   1. apt-get install nginx openssl if missing
#   2. Reads DASHBOARD_USER and DASHBOARD_PASS from .env, or prompts for them
#      (if DASHBOARD_PASS is missing, generates a random 20-char password and
#      prints it so you can copy it once)
#   3. Generates /etc/nginx/htpasswd-mila using openssl passwd -apr1
#   4. Generates a self-signed cert at /etc/ssl/mila-dashboard/ (valid 3 yrs,
#      CN=<ec2 hostname or localhost>) if one doesn't already exist
#   5. Writes /etc/nginx/sites-available/mila-dashboard from the template
#   6. Symlinks it into sites-enabled and removes the default site
#   7. Writes /etc/nginx/conf.d/mila-ratelimit.conf (the rate-limit zone)
#   8. Runs `nginx -t` and reloads nginx
#   9. Prints the public URL, the credentials (if generated), and a reminder
#      to open port 80/443 in the EC2 security group
#
# After running, the dashboard is reachable at:
#     https://<ec2-public-ip>/
# with the user/pass you set.
#
# Usage:
#     sudo ./nginx/install_nginx.sh
#     sudo ./nginx/install_nginx.sh --port 9000        # override upstream port
#     sudo ./nginx/install_nginx.sh --password mypass  # explicit password
#     sudo ./nginx/install_nginx.sh --user admin       # explicit user
#     sudo ./nginx/install_nginx.sh --no-https         # skip self-signed cert,
#                                                       # keep plain :80 auth
#
# Idempotent — safe to re-run. Overwrites config + htpasswd; keeps cert.

set -euo pipefail

# ---- paths -------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
TEMPLATE="$SCRIPT_DIR/mila-dashboard.conf.template"
SITE_NAME="mila-dashboard"
SITE_CONF="/etc/nginx/sites-available/${SITE_NAME}"
SITE_LINK="/etc/nginx/sites-enabled/${SITE_NAME}"
HTPASSWD="/etc/nginx/htpasswd-mila"
RATELIMIT_CONF="/etc/nginx/conf.d/mila-ratelimit.conf"
SSL_DIR="/etc/ssl/mila-dashboard"
SSL_CERT="$SSL_DIR/fullchain.pem"
SSL_KEY="$SSL_DIR/privkey.pem"

DEFAULT_UPSTREAM_PORT=8502
UPSTREAM_PORT="$DEFAULT_UPSTREAM_PORT"
USER_EXPLICIT=""
PASS_EXPLICIT=""
USE_HTTPS=1

# ---- color -------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
else
    C_RESET="" C_BOLD="" C_DIM="" C_RED="" C_GREEN="" C_YELLOW="" C_CYAN=""
fi

info()  { echo "${C_CYAN}▶${C_RESET} $*"; }
ok()    { echo "${C_GREEN}✓${C_RESET} $*"; }
warn()  { echo "${C_YELLOW}⚠${C_RESET} $*"; }
fail()  { echo "${C_RED}✗${C_RESET} $*" >&2; exit 1; }

# ---- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)      UPSTREAM_PORT="$2"; shift 2 ;;
        --user)      USER_EXPLICIT="$2"; shift 2 ;;
        --password)  PASS_EXPLICIT="$2"; shift 2 ;;
        --no-https)  USE_HTTPS=0; shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            fail "unknown flag: $1"
            ;;
    esac
done

# ---- must be root ------------------------------------------------------------
if [[ "$(id -u)" != "0" ]]; then
    fail "Please run as root (use sudo)."
fi

# ---- step 1: apt install nginx + openssl if missing --------------------------
info "Checking for nginx + openssl..."
MISSING_PKGS=()
command -v nginx  >/dev/null || MISSING_PKGS+=("nginx")
command -v openssl >/dev/null || MISSING_PKGS+=("openssl")
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing: ${MISSING_PKGS[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y "${MISSING_PKGS[@]}"
    ok "Packages installed"
else
    ok "nginx + openssl already present"
fi

# ---- step 2: resolve credentials ---------------------------------------------
DASHBOARD_USER="$USER_EXPLICIT"
DASHBOARD_PASS="$PASS_EXPLICIT"

# Try loading from .env if not given via flags
if [[ -f "$BASE_DIR/.env" ]]; then
    [[ -z "$DASHBOARD_USER" ]] && DASHBOARD_USER="$(grep -E '^DASHBOARD_USER=' "$BASE_DIR/.env" | cut -d= -f2- || true)"
    [[ -z "$DASHBOARD_PASS" ]] && DASHBOARD_PASS="$(grep -E '^DASHBOARD_PASS=' "$BASE_DIR/.env" | cut -d= -f2- || true)"
fi

# Defaults
[[ -z "$DASHBOARD_USER" ]] && DASHBOARD_USER="admin"

GENERATED_PASSWORD=0
if [[ -z "$DASHBOARD_PASS" ]]; then
    DASHBOARD_PASS="$(openssl rand -base64 18 | tr -d '+/=' | cut -c1-20)"
    GENERATED_PASSWORD=1
fi

# ---- step 3: htpasswd --------------------------------------------------------
info "Writing $HTPASSWD (user: $DASHBOARD_USER)"
HASH="$(openssl passwd -apr1 "$DASHBOARD_PASS")"
echo "${DASHBOARD_USER}:${HASH}" > "$HTPASSWD"
chmod 640 "$HTPASSWD"
chown root:www-data "$HTPASSWD" 2>/dev/null || true
ok "htpasswd written"

# ---- step 4: self-signed cert (if enabled) -----------------------------------
if [[ "$USE_HTTPS" == "1" ]]; then
    mkdir -p "$SSL_DIR"
    if [[ -f "$SSL_CERT" && -f "$SSL_KEY" ]]; then
        ok "Existing cert kept at $SSL_CERT"
    else
        info "Generating self-signed cert (valid 1095 days)..."
        CN="$(hostname -f 2>/dev/null || hostname || echo localhost)"
        openssl req -x509 -nodes \
            -days 1095 \
            -newkey rsa:2048 \
            -keyout "$SSL_KEY" \
            -out "$SSL_CERT" \
            -subj "/CN=$CN/O=MILA Test Control" \
            >/dev/null 2>&1
        chmod 640 "$SSL_KEY"
        chown root:www-data "$SSL_KEY" 2>/dev/null || true
        ok "Cert at $SSL_CERT (CN=$CN)"
    fi
fi

# ---- step 5: rate limit zone -------------------------------------------------
info "Writing $RATELIMIT_CONF"
cat > "$RATELIMIT_CONF" <<'EOF'
# Rate limit zone for the mila-dashboard site. Referenced by its limit_req
# directive. 10 MB of state tracks ~160k unique client IPs.
limit_req_zone $binary_remote_addr zone=mila_dashboard:10m rate=30r/s;
EOF
ok "rate limit zone written"

# ---- step 6: site config from template ---------------------------------------
info "Writing $SITE_CONF from template"

if [[ "$USE_HTTPS" == "1" ]]; then
    sed \
        -e "s|__UPSTREAM_PORT__|$UPSTREAM_PORT|g" \
        -e "s|__SSL_CERT_PATH__|$SSL_CERT|g" \
        -e "s|__SSL_KEY_PATH__|$SSL_KEY|g" \
        "$TEMPLATE" > "$SITE_CONF"
else
    # No-HTTPS variant: strip the :443 server block and un-redirect :80
    cat > "$SITE_CONF" <<EOF
# Generated by install_nginx.sh (--no-https mode)
limit_req_zone_referenced_by_site mila_dashboard;

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    server_tokens off;
    proxy_read_timeout  300s;
    proxy_send_timeout  300s;
    client_max_body_size 16m;

    limit_req zone=mila_dashboard burst=60 nodelay;

    location = /health {
        access_log off;
        auth_basic off;
        add_header Content-Type "text/plain";
        return 200 "ok\\n";
    }

    location / {
        auth_basic           "MILA Test Control";
        auth_basic_user_file $HTPASSWD;

        proxy_pass http://127.0.0.1:$UPSTREAM_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
    }
}
EOF
fi
ok "site config written"

# ---- step 7: enable + disable default + test + reload ------------------------
if [[ -L "/etc/nginx/sites-enabled/default" ]]; then
    info "Removing stock /etc/nginx/sites-enabled/default"
    rm -f /etc/nginx/sites-enabled/default
fi
ln -sfn "$SITE_CONF" "$SITE_LINK"

info "Testing nginx config (nginx -t)..."
if ! nginx -t >/dev/null 2>&1; then
    echo
    warn "nginx -t failed. Full output:"
    nginx -t || true
    fail "Configuration is invalid. Nothing reloaded."
fi
ok "nginx -t passed"

info "Reloading nginx..."
if systemctl is-active --quiet nginx; then
    systemctl reload nginx
else
    systemctl enable --now nginx >/dev/null 2>&1 || service nginx start
fi
ok "nginx reloaded"

# ---- optional: ufw ---------------------------------------------------------
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
    info "ufw is active — allowing 80/tcp and 443/tcp"
    ufw allow 80/tcp >/dev/null 2>&1 || true
    ufw allow 443/tcp >/dev/null 2>&1 || true
fi

# ---- step 8: final report ----------------------------------------------------
PUBLIC_IP="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<ec2-public-ip>')"

echo
echo "${C_BOLD}${C_GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo "${C_BOLD}${C_GREEN}  ✅ MILA dashboard nginx proxy installed ${C_RESET}"
echo "${C_BOLD}${C_GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo
if [[ "$USE_HTTPS" == "1" ]]; then
    echo "  URL:        ${C_BOLD}https://$PUBLIC_IP/${C_RESET}"
    echo "  (plain :80 redirects to https. Browsers will show a one-time"
    echo "   certificate warning because it's self-signed — click through.)"
else
    echo "  URL:        ${C_BOLD}http://$PUBLIC_IP/${C_RESET}"
    warn "No HTTPS — credentials are sent in plaintext on the wire."
fi
echo
echo "  Username:   ${C_BOLD}$DASHBOARD_USER${C_RESET}"
if [[ "$GENERATED_PASSWORD" == "1" ]]; then
    echo "  Password:   ${C_BOLD}$DASHBOARD_PASS${C_RESET}"
    echo "              ${C_YELLOW}↑ write this down, it was randomly generated${C_RESET}"
else
    echo "  Password:   (the one you provided)"
fi
echo
echo "${C_DIM}To change credentials later, re-run with --user and --password.${C_RESET}"
echo "${C_DIM}To remove everything: sudo ./nginx/uninstall_nginx.sh${C_RESET}"
echo
echo "${C_BOLD}⚠ AWS security group reminder:${C_RESET}"
echo "  Open inbound ${C_BOLD}TCP 443${C_RESET} (and optionally ${C_BOLD}TCP 80${C_RESET}) in the EC2 security group"
echo "  for the IPs of your stakeholders — or 0.0.0.0/0 if it's fine to be world-reachable"
echo "  (auth will still gate access)."
