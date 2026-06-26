"""
Mock backend. In-memory, deterministic, seeded with a handful of
customers/orders so eval fixtures can reference stable IDs.

This stands in for "the real systems" (order DB, payments, CRM) that a
production agent would call through real APIs. Refund/cancel state is
mutated here only when an approval is actually committed (see
approvals.py) -- never directly by the agent loop.
"""
from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    total_usd: float
    status: str  # "processing" | "shipped" | "delivered" | "cancelled"
    refunded_usd: float = 0.0


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    order_ids: list[str] = field(default_factory=list)


class MockBackend:
    """
    Thread-safe in-memory store. One instance shared across concurrent
    ticket workers within a process.
    """

    def __init__(self, *, failure_rate: float = 0.0, seed: int | None = 7):
        self._lock = threading.Lock()
        self._rng = random.Random(seed)
        self.failure_rate = failure_rate  # used by tools to simulate flaky externals
        self.customers: dict[str, Customer] = {}
        self.orders: dict[str, Order] = {}
        self._seed_data()

    def _seed_data(self) -> None:
        self.customers["cust_001"] = Customer(
            "cust_001", "Asha Rao", "asha@example.com", ["ord_1001", "ord_1002"]
        )
        self.customers["cust_002"] = Customer(
            "cust_002", "Ben Ortiz", "ben@example.com", ["ord_2001"]
        )
        self.customers["cust_003"] = Customer(
            "cust_003", "Priya Nair", "priya@example.com", ["ord_3001", "ord_3002"]
        )

        self.orders["ord_1001"] = Order("ord_1001", "cust_001", "Wireless Mouse", 24.99, "delivered")
        self.orders["ord_1002"] = Order("ord_1002", "cust_001", "USB-C Hub", 39.99, "shipped")
        self.orders["ord_2001"] = Order("ord_2001", "cust_002", "Mechanical Keyboard", 89.00, "delivered")
        self.orders["ord_3001"] = Order("ord_3001", "cust_003", "Monitor Arm", 54.50, "processing")
        self.orders["ord_3002"] = Order(
            "ord_3002", "cust_003", "Webcam", 45.00, "delivered", refunded_usd=45.00
        )  # already fully refunded -- used to test duplicate-refund denial

    def should_inject_failure(self) -> bool:
        return self._rng.random() < self.failure_rate

    def get_customer(self, customer_id: str) -> Customer | None:
        with self._lock:
            return self.customers.get(customer_id)

    def get_order(self, order_id: str) -> Order | None:
        with self._lock:
            return self.orders.get(order_id)

    def get_orders_for_customer(self, customer_id: str) -> list[Order]:
        with self._lock:
            cust = self.customers.get(customer_id)
            if not cust:
                return []
            return [self.orders[oid] for oid in cust.order_ids if oid in self.orders]

    def commit_refund(self, order_id: str, amount_usd: float) -> Order:
        with self._lock:
            order = self.orders[order_id]
            order.refunded_usd += amount_usd
            return order

    def commit_cancel(self, order_id: str) -> Order:
        with self._lock:
            order = self.orders[order_id]
            order.status = "cancelled"
            return order


KB_ARTICLES = [
    {
        "id": "kb_shipping_times",
        "title": "Standard shipping times",
        "body": "Standard shipping takes 3-5 business days domestically. "
                "Expedited shipping takes 1-2 business days.",
    },
    {
        "id": "kb_return_policy",
        "title": "Return and refund policy",
        "body": "Items may be returned within 30 days of delivery for a full refund. "
                "Refunds are issued to the original payment method and require human "
                "approval before funds are released.",
    },
    {
        "id": "kb_password_reset",
        "title": "Resetting your password",
        "body": "Customers can reset their password from the login page by selecting "
                "'Forgot password' and following the emailed link.",
    },
    {
        "id": "kb_order_tracking",
        "title": "Tracking your order",
        "body": "Order tracking links are emailed once an order ships. Tracking can "
                "take up to 24 hours to update after a label is created.",
    },
]
