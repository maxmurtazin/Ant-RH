from api.server import parse_aco_history_csv


def test_parse_aco_no_header_old_format(tmp_path):
    p = tmp_path / "aco.csv"
    p.write_text("0,88.5,88.6\n", encoding="utf-8")
    pts = parse_aco_history_csv(str(p), limit=300)
    assert len(pts) == 1
    assert pts[0]["iter"] == 0
    assert pts[0]["best_loss"] == 88.5
    assert pts[0]["mean_loss"] == 88.6
    assert pts[0]["best_reward"] is None
    assert pts[0]["mean_reward"] is None
    assert pts[0]["reward_mode"] is None


def test_parse_aco_no_header_new_format(tmp_path):
    p = tmp_path / "aco.csv"
    p.write_text("0,44.2,44.3,1.0,0.14,rank\n", encoding="utf-8")
    pts = parse_aco_history_csv(str(p), limit=300)
    assert len(pts) == 1
    assert pts[0]["iter"] == 0
    assert pts[0]["best_loss"] == 44.2
    assert pts[0]["mean_loss"] == 44.3
    assert pts[0]["best_reward"] == 1.0
    assert pts[0]["mean_reward"] == 0.14
    assert pts[0]["reward_mode"] == "rank"


def test_parse_aco_header_format(tmp_path):
    p = tmp_path / "aco.csv"
    p.write_text(
        "iter,best_loss,mean_loss,best_reward,mean_reward,reward_mode\n"
        "1,44,45,1,0.2,rank\n",
        encoding="utf-8",
    )
    pts = parse_aco_history_csv(str(p), limit=300)
    assert len(pts) == 1
    assert pts[0]["iter"] == 1
    assert pts[0]["best_loss"] == 44.0
    assert pts[0]["mean_loss"] == 45.0
    assert pts[0]["best_reward"] == 1.0
    assert pts[0]["mean_reward"] == 0.2
    assert pts[0]["reward_mode"] == "rank"

