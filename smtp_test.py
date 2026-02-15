import smtplib
import os
import traceback
from dotenv import load_dotenv
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

def send_test_email():
    print("--- SMTP Test Start ---")
    print(f"SENDER: {EMAIL_ADDRESS}")
    print(f"RECIPIENT: {RECIPIENT_EMAIL}")

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not RECIPIENT_EMAIL:
        print("Error: Missing environment variables. Please check .env file.")
        return

    msg = MIMEText("hello")
    msg['Subject'] = "SMTP TEST"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL

    try:
        # Use port 587 for STARTTLS
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        
        print("送信成功 (Sent Successfully)")

    except Exception as e:
        print("送信失敗 (Failed to send)")
        print(f"Exception: {e}")
        print("Full Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    send_test_email()
