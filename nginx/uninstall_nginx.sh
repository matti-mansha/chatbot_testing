#!/usr/bin/env bash
# uninstall_nginx.sh — clean removal of the dashboard's nginx config.
#
# What this does (sudo required):
#   * Removes /etc/nginx/sites-enabled/mila-dashboard
#   * Removes /etc/nginx/sites-available/mila-dashboard
#   * Removes /etc/nginx/conf.d/mila-ratelimit.conf
#   * Removes /etc/nginx/htpasswd-mila
#   * Re-enables the stock default site (if present)
#   * Reloads nginx
#
# Does NOT:
#   * Remove /etc/ssl/mila-dashboard/ (keep the cert in case you reinstall)
#   * Uninstall the nginx package itself
#
# After running, nginx still listens on :80 with the default welcome page
# (if you want to remove nginx entirely, run `sudo apt remove nginx`).

set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
    echo "Please run as root (use sudo)." >&2
    exit 1
fi

SITE_NAME="mila-dashboard"
SITE_CONF="/etc/nginx/sites-available/${SITE_NAME}"
SITE_LINK="/etc/nginx/sites-enabled/${SITE_NAME}"
HTPASSWD="/etc/nginx/htpasswd-mila"
RATELIMIT_CONF="/etc/nginx/conf.d/mila-ratelimit.conf"

removed_any=0

if [[ -L "$SITE_LINK" ]]; then
    rm -f "$SITE_LINK"
    echo "✓ removed $SITE_LINK"
    removed_any=1
fi

if [[ -f "$SITE_CONF" ]]; then
    rm -f "$SITE_CONF"
    echo "✓ removed $SITE_CONF"
    removed_any=1
fi

if [[ -f "$RATELIMIT_CONF" ]]; then
    rm -f "$RATELIMIT_CONF"
    echo "✓ removed $RATELIMIT_CONF"
    removed_any=1
fi

if [[ -f "$HTPASSWD" ]]; then
    rm -f "$HTPASSWD"
    echo "✓ removed $HTPASSWD"
    removed_any=1
fi

# Restore stock default site if it exists and isn't already enabled
if [[ -f "/etc/nginx/sites-available/default" ]] && [[ ! -L "/etc/nginx/sites-enabled/default" ]]; then
    ln -sfn /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default
    echo "✓ re-enabled stock /etc/nginx/sites-enabled/default"
fi

if [[ "$removed_any" == "0" ]]; then
    echo "Nothing to remove — mila-dashboard nginx config was not installed."
    exit 0
fi

echo
echo "Reloading nginx..."
if nginx -t >/dev/null 2>&1; then
    systemctl reload nginx 2>/dev/null || service nginx reload
    echo "✓ nginx reloaded"
else
    echo "⚠ nginx -t failed after removal — manual intervention may be needed:" >&2
    nginx -t || true
    exit 1
fi

echo
echo "All MILA nginx config removed. The self-signed cert at"
echo "/etc/ssl/mila-dashboard/ was kept for reinstallation."
