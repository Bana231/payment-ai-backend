from agent_graph import investigation_graph


clear_case_transactions = [
    {
        "transaction_id": "TEST1001",
        "amount": 100.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1003",
        "amount": 80.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1004",
        "amount": 210.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1005",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant E",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1006",
        "amount": 190.00,
        "currency": "USD",
        "merchant": "Merchant F",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
]


result = investigation_graph.invoke(
    {
        "question":
            "Why are these payment transactions failing?",

        "transactions":
            clear_case_transactions,
    }
)


print(result)