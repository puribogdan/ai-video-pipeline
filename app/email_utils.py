# app/email_utils.py
from __future__ import annotations
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

def _clean_ascii(val: str | None, field: str, strip_spaces: bool = False) -> str:
    if not val:
        raise RuntimeError(f"{field} is missing (set it in your Render env)")
    # normalize NBSP to regular space; trim ends
    val = val.replace("\u00a0", " ").strip()
    if strip_spaces:
        val = val.replace(" ", "")
    try:
        val.encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(f"{field} contains non-ASCII characters; please re-enter it in Render env")
    return val

def _get_from_fields() -> tuple[str, str]:
    """Return (display_name, email_address) for the From header."""
    from_env = os.getenv("FROM_EMAIL") or ""
    smtp_user = os.getenv("SMTP_USER") or ""
    display = ""
    addr = ""
    if "<" in from_env and ">" in from_env:
        # Format like: Your Name <you@example.com>
        try:
            display = from_env.split("<", 1)[0].strip().strip('"')
            addr = from_env.split("<", 1)[1].split(">", 1)[0].strip()
        except Exception:
            addr = from_env.strip()
    else:
        # Just an address or empty → fall back to SMTP_USER
        addr = from_env.strip() or smtp_user.strip()
    if not addr:
        raise RuntimeError("FROM_EMAIL/SMTP_USER not set")
    # Basic ascii safety for address only; display name can be unicode
    _ = _clean_ascii(addr, "FROM_EMAIL/SMTP_USER")
    return (display or addr, addr)

def _build_message_html(video_url: str, job_id: str) -> tuple[str, str]:
    # Make sure the URL has a scheme; if not, try to fix with APP_BASE_URL
    if not (video_url.startswith("http://") or video_url.startswith("https://")):
        base = os.getenv("APP_BASE_URL", "").rstrip("/")
        if base:
            video_url = f"{base}/media/{job_id}.mp4"

    subject = f"Your video is ready — job {job_id[:8]}"
    text = f"""\
Your video is ready!

Job ID: {job_id}
Link: {video_url}

If the button in the HTML version doesn’t work, copy/paste the link above.
"""

    html = f"""\
<!doctype html>
<html>
  <body style="font-family:Arial,Helvetica,sans-serif; line-height:1.5; color:#111">
    <p>Your video is ready!</p>
    <p><strong>Job ID:</strong> {job_id}</p>
    <p>
      <a href="{video_url}" target="_blank" rel="nofollow noopener noreferrer"
         style="display:inline-block;background:#1a73e8;color:#fff;text-decoration:none;
                padding:10px 16px;border-radius:6px;">
        Click here to view / download the video
      </a>
    </p>
    <p style="font-size:13px;color:#555;margin-top:16px">
      If the button doesn’t work, paste this link into your browser:<br>
      <a href="{video_url}" target="_blank" rel="nofollow noopener noreferrer">{video_url}</a>
    </p>
  </body>
</html>
"""
    return subject, text, html

def _send_via_smtp(to_email: str, subject: str, html: str, text: str) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = os.getenv("SMTP_TLS", "true").lower() == "true"

    # Sanitize credentials (avoid NBSP etc.)
    smtp_user = _clean_ascii(os.getenv("SMTP_USER"), "SMTP_USER")
    smtp_pass = _clean_ascii(os.getenv("SMTP_PASSWORD"), "SMTP_PASSWORD", strip_spaces=True)

    from_display, from_addr = _get_from_fields()

    # Build a proper multipart/alternative email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((from_display, from_addr))
    msg["To"] = to_email

    part_text = MIMEText(text, "plain", "utf-8")
    part_html = MIMEText(html, "html", "utf-8")
    msg.attach(part_text)
    msg.attach(part_html)

    context = ssl.create_default_context()
    server = smtplib.SMTP(host, port, timeout=30)
    try:
        if use_tls:
            server.starttls(context=context)
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_addr, [to_email], msg.as_string())
        print(f"[email] sent to {to_email} ({len(html)} html chars)", flush=True)
    finally:
        try:
            server.quit()
        except Exception:
            pass

def send_link_email(to_email: str, video_url: str, job_id: str) -> None:
    subject, text, html = _build_message_html(video_url, job_id)

    # If using Resend, prefer API (optional)
    resend_key = os.getenv("RESEND_API_KEY", "").strip()
    if resend_key:
        try:
            import resend  # type: ignore
            resend.api_key = resend_key
            from_display, from_addr = _get_from_fields()
            resend.Emails.send({
                "from": f"{from_display} <{from_addr}>",
                "to": [to_email],
                "subject": subject,
                "html": html,
                "text": text,
            })
            print(f"[email] sent via Resend to {to_email}", flush=True)
            return
        except Exception as e:
            print(f"[email] Resend failed, falling back to SMTP: {e}", flush=True)

    _send_via_smtp(to_email, subject, html, text)
