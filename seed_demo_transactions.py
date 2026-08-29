"""Seed Supabase with the project's existing synthetic transaction fixtures."""

from data import transactions
from supabase_store import get_supabase


def main() -> None:
    demo_transactions = [
        {
            **transaction,
            "environment": "sandbox",
        }
        for transaction in transactions
    ]

    response = (
        get_supabase()
        .table("transactions")
        .upsert(demo_transactions, on_conflict="transaction_id")
        .execute()
    )

    print(f"Seeded {len(response.data)} synthetic transactions.")


if __name__ == "__main__":
    main()
