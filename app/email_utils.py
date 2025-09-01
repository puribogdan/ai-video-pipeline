# app/email_utils.py
import os
import smtplib
from email.message import EmailMessage

import resend
from dotenv import load_dotenv

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "no-reply@example.com")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_TLS = os.getenv("SMTP_TLS", "true").lower() == "true"


def _send_via_resend(to_email: str, subject: str, html: str) -> None:
    resend.api_key = RESEND_API_KEY
    resend.Emails.send(
        {
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "html": html,
        }
    )


def _send_via_smtp(to_email: str, subject: str, html: str) -> None:
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD]):
        raise RuntimeError("SMTP not configured. Missing host/port/user/password.")

    msg = EmailMessage()
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(
        "Your video is ready! Open the link above.\n"
        "If you cannot see the link, please reply to this email."
    )
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        if SMTP_TLS:
            server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


def send_link_email(to_email: str, video_url: str, job_id: str) -> None:
    subject = f"Your video is ready (job {job_id})"
    html = f"""
    <p>Your video is ready! 🎉</p>
    <p><a href="{video_url}">Click here to view/download the video</a></p>
    <p>Job ID: <code>{job_id}</code></p>
    """
    if RESEND_API_KEY:
        _send_via_resend(to_email, subject, html)
    else:
        _send_via_smtp(to_email, subject, html)
