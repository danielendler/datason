#!/usr/bin/env python3
"""
Comprehensive Data Integrity & Verification Demo
===============================================

This example demonstrates datason's integrity features for:
- Compliance workflows and audit trails
- ML model verification and tamper detection
- Data pipeline integrity checks
- Digital signatures for high-security scenarios

Real-world use cases:
- Financial data processing (SOX compliance)
- Healthcare data (HIPAA audit trails)
- ML model deployment verification
- Contract and document integrity
"""

import uuid
from datetime import datetime

import datason


def demonstrate_basic_integrity():
    """Basic integrity verification for audit trails."""
    print("=" * 60)
    print("🔐 BASIC INTEGRITY VERIFICATION")
    print("=" * 60)

    # Sample transaction data
    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(),
        "amount": 1500.00,
        "account_from": "ACC123456",
        "account_to": "ACC789012",
        "description": "Monthly payment",
    }

    # 1. Compute hash for audit trail
    original_hash = datason.hash_object(transaction)
    print(f"📊 Transaction Hash: {original_hash}")

    # 2. Verify data hasn't been tampered with
    is_valid = datason.verify_object(transaction, original_hash)
    print(f"✅ Data Integrity: {'VALID' if is_valid else 'COMPROMISED'}")

    # 3. Detect tampering
    tampered_transaction = transaction.copy()
    tampered_transaction["amount"] = 15000.00  # Someone changed the amount!

    is_tampered = datason.verify_object(tampered_transaction, original_hash)
    print(f"🚨 Tampered Data: {'VALID' if is_tampered else 'DETECTED TAMPERING'}")

    print("\n💡 Use Case: Financial audit trails, regulatory compliance")


def demonstrate_redaction_compliance():
    """GDPR/HIPAA compliance with redaction-aware hashing."""
    print("\n" + "=" * 60)
    print("🏥 COMPLIANCE & REDACTION WORKFLOWS")
    print("=" * 60)

    # Sample patient data (HIPAA scenario)
    patient_data = {
        "patient_id": "P123456",
        "name": "John Doe",
        "ssn": "123-45-6789",
        "email": "john.doe@email.com",
        "diagnosis": "Type 2 Diabetes",
        "treatment_plan": "Metformin, lifestyle changes",
        "doctor": "Dr. Smith",
        "visit_date": datetime.now(),
    }

    # 1. Hash original data (for internal audit)
    original_hash = datason.hash_object(patient_data)
    print(f"📊 Original Data Hash: {original_hash}")

    try:
        # 2. Create HIPAA-compliant redacted version
        redacted_data, redacted_hash = datason.hash_and_redact(
            patient_data,
            redact={
                "fields": ["ssn", "email", "name"],  # Remove PII
                "replacement": "[REDACTED-HIPAA]",
            },
        )

        print(f"🔒 Redacted Data Hash: {redacted_hash}")
        print("🏥 Redacted Patient Record:")
        for key, value in redacted_data.items():
            print(f"    {key}: {value}")

        # 3. Verify redacted data integrity
        is_redacted_valid = datason.verify_object(redacted_data, redacted_hash)
        print(f"✅ Redacted Data Integrity: {'VALID' if is_redacted_valid else 'COMPROMISED'}")

        # 4. Demonstrate reproducible redaction hashing
        _, same_redacted_hash = datason.hash_and_redact(
            patient_data, redact={"fields": ["ssn", "email", "name"], "replacement": "[REDACTED-HIPAA]"}
        )

        hashes_match = redacted_hash == same_redacted_hash
        print(f"🔄 Reproducible Redaction: {'YES' if hashes_match else 'NO'}")

        print("\n💡 Use Case: HIPAA compliance, GDPR right to be forgotten")

    except RuntimeError as e:
        print(f"⚠️  Redaction not available: {e}")
        print("💡 Install with: pip install datason[all]")


def demonstrate_ml_model_verification():
    """ML model integrity for deployment pipelines."""
    print("\n" + "=" * 60)
    print("🤖 ML MODEL VERIFICATION PIPELINE")
    print("=" * 60)

    # Simulate ML model metadata
    model_metadata = {
        "model_name": "fraud_detection_v2.1",
        "training_data_hash": "sha256:abc123...",
        "model_params": {"learning_rate": 0.001, "epochs": 50, "batch_size": 32},
        "accuracy_metrics": {"precision": 0.94, "recall": 0.92, "f1_score": 0.93},
        "training_timestamp": datetime.now(),
        "data_scientist": "alice@company.com",
        "deployment_approved": True,
    }

    # 1. Development team creates model hash
    dev_hash = datason.hash_object(model_metadata)
    print(f"🔬 Development Hash: {dev_hash}")

    # 2. QA team verifies model hasn't changed
    qa_verification = datason.verify_object(model_metadata, dev_hash)
    print(f"🧪 QA Verification: {'PASSED' if qa_verification else 'FAILED'}")

    # 3. Production deployment verification
    production_verification = datason.verify_object(model_metadata, dev_hash)
    print(f"🚀 Production Verification: {'SAFE TO DEPLOY' if production_verification else 'DEPLOYMENT BLOCKED'}")

    # 4. Simulate model drift detection
    updated_metadata = model_metadata.copy()
    updated_metadata["accuracy_metrics"]["f1_score"] = 0.85  # Performance degraded

    drift_detected = not datason.verify_object(updated_metadata, dev_hash)
    print(f"📉 Model Drift Detection: {'DRIFT DETECTED' if drift_detected else 'MODEL STABLE'}")

    print("\n💡 Use Case: MLOps pipelines, model governance, deployment safety")


def demonstrate_digital_signatures():
    """High-security scenarios with digital signatures."""
    print("\n" + "=" * 60)
    print("🔏 DIGITAL SIGNATURES (HIGH SECURITY)")
    print("=" * 60)

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        # Generate key pair (in practice, load from secure storage)
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        # High-value contract data
        contract = {
            "contract_id": "CONTRACT-2024-001",
            "parties": ["Company A", "Company B"],
            "amount": 10000000.00,  # $10M contract
            "terms": "Software licensing agreement",
            "effective_date": datetime.now(),
            "duration_years": 3,
            "signed_by": "legal@companyA.com",
        }

        # 1. Sign the contract
        signature = datason.sign_object(contract, private_pem)
        print(f"✍️  Digital Signature: {signature[:50]}...")

        # 2. Verify signature authenticity
        is_authentic = datason.verify_signature(contract, signature, public_pem)
        print(f"🔍 Signature Verification: {'AUTHENTIC' if is_authentic else 'FORGED'}")

        # 3. Detect tampering with signed document
        tampered_contract = contract.copy()
        tampered_contract["amount"] = 100000000.00  # Someone added a zero!

        tampered_verification = datason.verify_signature(tampered_contract, signature, public_pem)
        print(f"🚨 Tampered Document: {'VALID' if tampered_verification else 'FORGERY DETECTED'}")

        # 4. Redaction-aware signing (for partially public contracts)
        try:
            redacted_signature = datason.sign_object(
                contract, private_pem, redact={"fields": ["amount"], "replacement": "[CONFIDENTIAL]"}
            )

            # Verify redacted version
            redacted_verification = datason.verify_signature(
                contract, redacted_signature, public_pem, redact={"fields": ["amount"], "replacement": "[CONFIDENTIAL]"}
            )
            print(f"🤐 Redacted Signature: {'VALID' if redacted_verification else 'INVALID'}")

        except RuntimeError:
            print("⚠️  Redaction not available for signature example")

        print("\n💡 Use Case: Legal contracts, financial instruments, regulatory filings")

    except ImportError:
        print("⚠️  Cryptography package not installed")
        print("💡 Install with: pip install datason[crypto]")
        print("🔐 Digital signatures require the cryptography package for Ed25519 support")


def demonstrate_audit_trail_workflow():
    """Complete audit trail workflow for compliance."""
    print("\n" + "=" * 60)
    print("📋 COMPLETE AUDIT TRAIL WORKFLOW")
    print("=" * 60)

    # Simulate a data processing pipeline
    pipeline_steps = []

    # Step 1: Data ingestion
    raw_data = {
        "data_source": "customer_database",
        "records_count": 10000,
        "extraction_time": datetime.now(),
        "checksum": "md5:abc123def456",
    }

    step1_hash = datason.hash_object(raw_data)
    pipeline_steps.append(
        {"step": "data_ingestion", "data_hash": step1_hash, "timestamp": datetime.now(), "status": "completed"}
    )
    print(f"1️⃣  Data Ingestion: {step1_hash}")

    # Step 2: Data cleaning
    cleaned_data = raw_data.copy()
    cleaned_data.update(
        {
            "records_after_cleaning": 9850,
            "removed_duplicates": 150,
            "cleaning_rules_applied": ["remove_nulls", "deduplicate", "validate_emails"],
        }
    )

    step2_hash = datason.hash_object(cleaned_data)
    pipeline_steps.append(
        {"step": "data_cleaning", "data_hash": step2_hash, "timestamp": datetime.now(), "status": "completed"}
    )
    print(f"2️⃣  Data Cleaning: {step2_hash}")

    # Step 3: Feature engineering
    featured_data = cleaned_data.copy()
    featured_data.update(
        {
            "features_created": ["customer_lifetime_value", "churn_risk_score"],
            "feature_count": 45,
            "feature_importance": {"clv": 0.8, "churn": 0.6},
        }
    )

    step3_hash = datason.hash_object(featured_data)
    pipeline_steps.append(
        {"step": "feature_engineering", "data_hash": step3_hash, "timestamp": datetime.now(), "status": "completed"}
    )
    print(f"3️⃣  Feature Engineering: {step3_hash}")

    # Create audit trail hash
    audit_trail = {
        "pipeline_id": str(uuid.uuid4()),
        "pipeline_version": "1.2.0",
        "execution_date": datetime.now(),
        "steps": pipeline_steps,
        "total_execution_time": "45 minutes",
        "data_scientist": "bob@company.com",
        "approved_by": "alice@company.com",
    }

    audit_hash = datason.hash_object(audit_trail)
    print(f"📋 Complete Audit Trail: {audit_hash}")

    # Verify entire pipeline
    pipeline_verification = datason.verify_object(audit_trail, audit_hash)
    print(f"✅ Pipeline Integrity: {'VERIFIED' if pipeline_verification else 'COMPROMISED'}")

    # Save audit trail (simulation)
    # In production, you would save this to a database or audit log
    print("\n📄 Audit Record Saved for Compliance")
    print(f"🔍 Verification Hash: {audit_hash}")
    print("📋 Compliance Status: VERIFIED")
    print("🏢 Frameworks: SOX, GDPR, CCPA")
    print("💡 Use Case: Data pipeline auditing, regulatory compliance, change tracking")


def demonstrate_hash_algorithms():
    """Demonstrate different hash algorithms for various security needs."""
    print("\n" + "=" * 60)
    print("🧮 HASH ALGORITHM COMPARISON")
    print("=" * 60)

    test_data = {
        "sensitive_data": "highly_confidential_information",
        "timestamp": datetime.now(),
        "classification": "TOP_SECRET",
    }

    algorithms = ["sha256", "sha1", "md5", "sha512"]

    print("🔐 Same data, different hash algorithms:")
    for algo in algorithms:
        try:
            hash_result = datason.hash_object(test_data, hash_algo=algo)
            security_level = {
                "sha256": "🔒 RECOMMENDED",
                "sha512": "🛡️  HIGH SECURITY",
                "sha1": "⚠️  DEPRECATED",
                "md5": "❌ INSECURE",
            }.get(algo, "❓ UNKNOWN")

            print(f"  {algo.upper()}: {hash_result} {security_level}")
        except Exception as e:
            print(f"  {algo.upper()}: ❌ Error - {e}")

    print("\n💡 Recommendation: Use SHA256 (default) or SHA512 for production")


def main():
    """Run all integrity verification demonstrations."""
    print("🔐 DATASON INTEGRITY VERIFICATION COMPREHENSIVE DEMO")
    print("=" * 60)
    print("Demonstrating real-world use cases for data integrity and verification\n")

    demonstrate_basic_integrity()
    demonstrate_redaction_compliance()
    demonstrate_ml_model_verification()
    demonstrate_digital_signatures()
    demonstrate_audit_trail_workflow()
    demonstrate_hash_algorithms()

    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("📚 Key Takeaways:")
    print("  • Use hash_object() for basic integrity verification")
    print("  • Use hash_and_redact() for GDPR/HIPAA compliance")
    print("  • Use digital signatures for high-security scenarios")
    print("  • Build complete audit trails for regulatory compliance")
    print("  • Choose appropriate hash algorithms for your security needs")
    print("\n🔗 Learn More:")
    print("  • Documentation: docs/integrity.md")
    print("  • API Reference: datason.integrity module")
    print("  • Production Examples: examples/framework_integrations/")


if __name__ == "__main__":
    main()
