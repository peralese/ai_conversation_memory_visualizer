from src.redaction.redactor import RedactionConfig, redact_text


def test_redact_email_phone_ip():
    text = "Contact me at me@example.com or +1 555-123-4567 from 10.0.0.1"
    out = redact_text(
        text,
        RedactionConfig(redact_emails=True, redact_phones=True, redact_ips=True, redact_names=False),
    )

    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_IP]" in out
