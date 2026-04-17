"""PII sanitizer tests — regex, LLM, and hybrid modes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from loci_extract.sanitizer import sanitize, sanitize_regex

# ---------------------------------------------------------------------------
# Regex mode
# ---------------------------------------------------------------------------


def test_regex_replaces_ssn():
    text = "Employee SSN: 123-45-6789. Another: 987-65-4321."
    result = sanitize_regex(text)
    assert "123-45-6789" not in result["sanitized"]
    assert "987-65-4321" not in result["sanitized"]
    # Should have synthetic SSNs (900+ area range)
    assert result["mode"] == "regex"
    assert len(result["replacements"]) >= 2
    for r in result["replacements"]:
        if r["kind"] == "ssn":
            assert r["replacement"].startswith("9")


def test_regex_preserves_ein():
    text = "Employer EIN: 12-3456789. Employee SSN: 123-45-6789."
    result = sanitize_regex(text)
    # EIN should NOT be touched
    assert "12-3456789" in result["sanitized"]
    # SSN should be replaced
    assert "123-45-6789" not in result["sanitized"]


def test_regex_replaces_phone():
    text = "Call (415) 555-1234 or 650-123-4567."
    result = sanitize_regex(text)
    assert "(415) 555-1234" not in result["sanitized"]
    phones = [r for r in result["replacements"] if r["kind"] == "phone"]
    assert len(phones) >= 1


def test_regex_consistent_replacement():
    """Same PII value appearing twice should get the same synthetic value."""
    text = "SSN: 123-45-6789. Again: 123-45-6789."
    result = sanitize_regex(text)
    # Only 1 replacement entry (deduped)
    ssn_replacements = [r for r in result["replacements"] if r["kind"] == "ssn"]
    assert len(ssn_replacements) == 1
    # Both occurrences replaced with the same synthetic
    synthetic = ssn_replacements[0]["replacement"]
    assert result["sanitized"].count(synthetic) == 2


def test_regex_deterministic():
    """Same input should produce same output (hash-based, not random)."""
    text = "SSN: 111-22-3333"
    r1 = sanitize_regex(text)
    r2 = sanitize_regex(text)
    assert r1["sanitized"] == r2["sanitized"]


# ---------------------------------------------------------------------------
# LLM mode (stubbed)
# ---------------------------------------------------------------------------


def test_llm_mode_calls_client():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Sanitized output from LLM"))],
        usage=MagicMock(prompt_tokens=100, completion_tokens=50),
    )
    result = sanitize("SSN: 123-45-6789", mode="llm", client=mock_client, model_name="test-model")
    assert result["mode"] == "llm"
    assert result["sanitized"] == "Sanitized output from LLM"
    assert result["model"] == "test-model"
    mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Hybrid mode (stubbed)
# ---------------------------------------------------------------------------


def test_hybrid_runs_regex_then_llm():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Fully sanitized text"))],
        usage=MagicMock(prompt_tokens=100, completion_tokens=50),
    )
    result = sanitize("SSN: 123-45-6789. Name: Jane Smith.",
                      mode="hybrid", client=mock_client, model_name="test-model")
    assert result["mode"] == "hybrid"
    assert result["sanitized"] == "Fully sanitized text"
    # Regex replacements should be recorded
    assert "regex_replacements" in result
    # The LLM should have been called (on the regex-sanitized text)
    mock_client.chat.completions.create.assert_called_once()
    # The call should NOT contain the original SSN (regex removed it first)
    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args.kwargs["messages"][-1]["content"]
    assert "123-45-6789" not in user_msg


# ---------------------------------------------------------------------------
# Dispatch validation
# ---------------------------------------------------------------------------


def test_sanitize_raises_on_unknown_mode():
    with pytest.raises(ValueError, match="Unknown sanitize mode"):
        sanitize("text", mode="unknown")


def test_sanitize_raises_without_client_for_llm():
    with pytest.raises(ValueError, match="LLM client required"):
        sanitize("text", mode="llm")
