from homework2_rent import score_rent, download_clean_data


def test_rent():
    X, y = download_clean_data()
    r2 = score_rent(X, y)
    assert( r2 > 0.29)


