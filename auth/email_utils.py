import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "1243hassan@gmail.com"
EMAIL_PASSWORD = "duqkeoraqfukoypj"  # NOT your real password


def send_welcome_email(to_email: str, first_name: str, last_name: str):
    subject = "Welcome to AutoPrepAI 🚀"

    body = f"""
Hi {first_name} {last_name},

Welcome to AutoPrepAI!

We're excited to have you on board. You can now start cleaning, processing, and optimizing your datasets with ease.

If you have any questions, feel free to explore the platform or reach out.

Happy building! 🚀

— AutoPrepAI Team
"""

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Email sending failed: {e}")