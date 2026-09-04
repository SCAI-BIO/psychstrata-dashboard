from datetime import date, datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def age_on_date(date_of_birth: date, on_date: date | None = None) -> int:
    effective_date = on_date or date.today()
    return effective_date.year - date_of_birth.year - (
        (effective_date.month, effective_date.day) < (date_of_birth.month, date_of_birth.day)
    )
