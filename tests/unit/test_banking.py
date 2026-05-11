"""Tests for Enterprise Banking Domain (CONCEPT:KG-2.8)."""

from agent_utilities.domains.finance.banking_models import (
    AMLAlertNode,
    AMLAlertSeverity,
    BankAccountNode,
    CreditRiskNode,
    ISO20022MessageType,
    KYCRecordNode,
    KYCRiskLevel,
    PaymentMessageNode,
    RegulatoryCapitalNode,
    SettlementCycle,
    SettlementLegNode,
)
from agent_utilities.domains.finance.banking import (
    ISO20022MessageFactory,
    KYCAMLEngine,
    RegulatoryCapitalCalculator,
    SettlementEngine,
)


class TestBankingModels:
    def test_bank_account(self):
        acc = BankAccountNode(
            id="acc:001",
            account_name="Operating",
            iban="DE89370400440532013000",
            swift_code="COBADEFFXXX",
        )
        assert acc.iban.startswith("DE")

    def test_payment_message(self):
        msg = PaymentMessageNode(
            id="msg:001",
            message_type=ISO20022MessageType.PACS_008,
            sender_bic="COBADEFFXXX",
            receiver_bic="DEUTDEFFXXX",
            amount=50000.0,
        )
        assert msg.message_type == ISO20022MessageType.PACS_008

    def test_kyc_record(self):
        kyc = KYCRecordNode(
            id="kyc:001",
            customer_id="cust:001",
            risk_classification=KYCRiskLevel.HIGH,
            pep_status=True,
        )
        assert kyc.pep_status is True

    def test_credit_risk(self):
        cr = CreditRiskNode(
            id="cr:001",
            account_id="acc:001",
            probability_of_default=0.05,
            loss_given_default=0.45,
            exposure_at_default=1000000.0,
        )
        assert cr.probability_of_default == 0.05

    def test_regulatory_capital(self):
        rc = RegulatoryCapitalNode(
            id="rc:001",
            cet1_ratio=0.12,
            total_capital_ratio=0.16,
            risk_weighted_assets=5000000.0,
        )
        assert rc.cet1_ratio == 0.12


class TestISO20022Factory:
    def test_credit_transfer(self):
        msg = ISO20022MessageFactory.create_credit_transfer(
            sender_bic="COBADEFFXXX",
            receiver_bic="DEUTDEFFXXX",
            amount=100000.0,
            currency="EUR",
        )
        assert msg.message_type == ISO20022MessageType.PACS_008
        assert msg.amount == 100000.0
        assert msg.end_to_end_id.startswith("E2E-")

    def test_payment_initiation(self):
        msg = ISO20022MessageFactory.create_payment_initiation(
            sender_bic="COBADEFFXXX",
            amount=50000.0,
        )
        assert msg.message_type == ISO20022MessageType.PAIN_001


class TestKYCAMLEngine:
    def test_no_alert_below_threshold(self):
        engine = KYCAMLEngine(structuring_threshold=10000.0)
        alert = engine.check_transaction("tx:001", "acc:001", 5000.0)
        assert alert is None

    def test_structuring_alert(self):
        engine = KYCAMLEngine(structuring_threshold=10000.0)
        alert = engine.check_transaction("tx:002", "acc:001", 9500.0)
        assert alert is not None
        assert alert.alert_type == "structuring"

    def test_sar_required(self):
        engine = KYCAMLEngine(structuring_threshold=10000.0)
        alert = engine.check_transaction("tx:003", "acc:001", 15000.0)
        assert alert is not None
        assert alert.severity == AMLAlertSeverity.SAR_REQUIRED

    def test_get_unresolved_alerts(self):
        engine = KYCAMLEngine(structuring_threshold=10000.0)
        engine.check_transaction("tx:001", "acc:001", 15000.0)
        alerts = engine.get_alerts(account_id="acc:001")
        assert len(alerts) == 1


class TestSettlementEngine:
    def test_settlement_chain(self):
        engine = SettlementEngine()
        legs = engine.create_settlement_chain(
            payment_id="pay:001",
            correspondent_chain=["bank:A", "bank:B", "bank:C"],
            amount=1000000.0,
            cycle=SettlementCycle.T_PLUS_1,
        )
        assert len(legs) == 3
        assert legs[0].leg_sequence == 1
        assert legs[2].leg_sequence == 3

    def test_settlement_status(self):
        engine = SettlementEngine()
        engine.create_settlement_chain(
            payment_id="pay:002",
            correspondent_chain=["bank:X"],
            amount=500000.0,
        )
        status = engine.get_settlement_status("pay:002")
        assert len(status) == 1


class TestRegulatoryCapital:
    def test_calculate_ratios(self):
        rc = RegulatoryCapitalCalculator.calculate_ratios(
            cet1_capital=450000,
            at1_capital=100000,
            tier2_capital=200000,
            risk_weighted_assets=5000000,
        )
        assert rc.cet1_ratio == 0.09  # 450k/5M = 9%
        assert rc.total_capital_ratio == 0.15  # 750k/5M = 15%

    def test_compliance_check_pass(self):
        rc = RegulatoryCapitalCalculator.calculate_ratios(
            cet1_capital=450000,
            at1_capital=100000,
            tier2_capital=200000,
            risk_weighted_assets=5000000,
        )
        check = RegulatoryCapitalCalculator.check_compliance(rc)
        assert check["cet1_compliant"] is True
        assert check["total_capital_compliant"] is True

    def test_compliance_check_fail(self):
        rc = RegulatoryCapitalCalculator.calculate_ratios(
            cet1_capital=20000,
            at1_capital=10000,
            tier2_capital=10000,
            risk_weighted_assets=5000000,
        )
        check = RegulatoryCapitalCalculator.check_compliance(rc)
        assert check["cet1_compliant"] is False
        assert check["total_capital_compliant"] is False
