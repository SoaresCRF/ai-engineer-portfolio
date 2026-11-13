from review_triage.report import gpu_usd


def test_gpu_usd_prices_one_t4_hour_at_the_reference_rate():
    assert gpu_usd(3_600) == 0.35
    # 1,000 reviews at 0.71 reviews/s (the teacher) cost about 14 cents.
    assert gpu_usd(1_000 / 0.71) == 0.1369
