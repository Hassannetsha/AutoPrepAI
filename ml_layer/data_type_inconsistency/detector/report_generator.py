class ReportGenerator:
    def generate(self, results):
        if not results:
            return "No analysis results available. Run analyze_dataframe() first."

        report = ["=" * 80, "DATA TYPE INCONSISTENCY DETECTION REPORT", "=" * 80]

        for col, result in results.items():
            report.append(f"\nColumn: {col}")
            report.append("-" * 80)
            report.append(f"  Declared dtype: {result['declared_dtype']}")
            report.append(f"  Total rows: {result['total_rows']}")
            report.append(f"  Null count: {result['null_count']} ({result['null_count']/result['total_rows']*100:.1f}%)")
            report.append(f"  Recommended type: {result['recommended_type']}")

            report.append(f"\n  Detected types:")
            for dtype, count in result['detected_types'].items():
                pct = count / result['non_null_count'] * 100 if result['non_null_count'] > 0 else 0
                report.append(f"    - {dtype}: {count} ({pct:.1f}%)")

            if result['inconsistencies']:
                report.append(f"\n  ⚠️  INCONSISTENCIES FOUND:")
                for issue in result['inconsistencies']:
                    report.append(f"    - {issue}")

            if result['conversion_issues']:
                report.append(f"\n  ⚠️  CONVERSION ISSUES:")
                for issue in result['conversion_issues']:
                    report.append(f"    - {issue}")

            if not result['inconsistencies'] and not result['conversion_issues']:
                report.append(f"\n  ✓ No inconsistencies detected")
            
            if result.get('inconsistent_indices'):
                report.append(f"\n  ⚠️  {len(result['inconsistent_indices'])} inconsistent rows identified.")

        report.append("\n" + "=" * 80)
        return "\n".join(report)
