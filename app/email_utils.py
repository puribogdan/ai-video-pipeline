import os
from typing import Optional


import resend
from dotenv import load_dotenv


load_dotenv()


RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "no-reply@example.com")


# Optional SMTP fallback (not implemented here, but wired for extension)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_TLS = os.getenv("SMTP_TLS", "true").lower() == "true"




def send_link_email(to_email: str, video_url: str, job_id: str) -> None:
"""Send a simple email with a link to the finished video.
Uses Resend if configured; you can add SMTP fallback if desired.
"""
subject = f"Your video is ready (job {job_id})"
html = f"""
<p>Your video is ready! 🎉</p>
<p><a href="{video_url}">Click here to view/download the video</a></p>
<p>Job ID: <code>{job_id}</code></p>
"""


if RESEND_API_KEY:
resend.api_key = RESEND_API_KEY
resend.Emails.send(
{
"from": FROM_EMAIL,
"to": [to_email],
"subject": subject,
"html": html,
}
)
return


# TODO: Implement SMTP fallback if needed
raise RuntimeError("No email provider configured. Set RESEND_API_KEY or implement SMTP fallback.")