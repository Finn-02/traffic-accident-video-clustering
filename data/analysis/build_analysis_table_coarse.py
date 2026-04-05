#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

import pandas as pd


DEFAULT_EXPERIMENT = "moco3"
DEFAULT_MODEL = "ViViT"
DEFAULT_EPOCH = 100

BASE_FINE_OUTPUT_DIR = Path("./VideoContrastive/data/analysis/analysis_output")
BASE_COARSE_OUTPUT_DIR = Path("./VideoContrastive/data/analysis/analysis_output_coarse")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build coarse analysis table from merged fine analysis table.")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=["moco3", "simclr"])
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["ViViT", "X3D", "C3D"])
    p.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    p.add_argument("--input-csv", type=str, default=None,
                   help="Optional override path to merged_analysis_table.csv")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Optional override coarse output directory")
    return p.parse_args()


def experiment_tag(experiment: str, model: str, epoch: int) -> str:
    return f"{experiment}_{model}_{epoch}"


def expand_ranges(*parts: Any) -> List[int]:
    values: List[int] = []
    for part in parts:
        if isinstance(part, int):
            values.append(part)
        elif isinstance(part, tuple) and len(part) == 2:
            start, end = part
            values.extend(list(range(start, end + 1)))
        elif isinstance(part, list):
            values.extend(expand_ranges(*part))
        else:
            raise ValueError(f"Unsupported range element: {part}")
    return values


def invert_group_dict(group_to_codes: Dict[str, List[int]]) -> Tuple[Dict[int, str], List[Dict[str, Any]], List[int]]:
    code_to_group: Dict[int, str] = {}
    duplicates: List[Dict[str, Any]] = []
    all_codes: List[int] = []

    for group_name, codes in group_to_codes.items():
        for code in codes:
            all_codes.append(code)
            if code in code_to_group:
                duplicates.append({
                    "fine_code": code,
                    "first_group": code_to_group[code],
                    "duplicate_group": group_name,
                })
            else:
                code_to_group[code] = group_name

    return code_to_group, duplicates, sorted(set(all_codes))


def build_coarse_codebook(group_names: List[str], include_others: bool = True) -> Dict[str, int]:
    names = list(group_names)
    if include_others and "Others" not in names:
        names.append("Others")
    return {name: idx for idx, name in enumerate(names)}


def apply_mapping(series: pd.Series, code_to_group: Dict[int, str], group_to_coarse_code: Dict[str, int]) -> pd.Series:
    def mapper(x):
        try:
            ix = int(x)
        except Exception:
            return group_to_coarse_code["Others"]
        group = code_to_group.get(ix, "Others")
        return group_to_coarse_code[group]

    return series.map(mapper).astype(int)


def save_dictionary_csv(path: Path, coarse_codebook: Dict[str, int]) -> None:
    rows = [{"coarse_code": code, "coarse_name": name} for name, code in coarse_codebook.items()]
    rows = sorted(rows, key=lambda x: x["coarse_code"])
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_fine_to_coarse_mapping(
    path: Path,
    used_fine_codes: Iterable[int],
    code_to_group: Dict[int, str],
    coarse_codebook: Dict[str, int],
) -> None:
    rows = []
    for fine_code in sorted(set(int(c) for c in used_fine_codes if pd.notna(c))):
        group_name = code_to_group.get(int(fine_code), "Others")
        rows.append({
            "fine_code": int(fine_code),
            "coarse_code": int(coarse_codebook[group_name]),
            "coarse_name": group_name,
            "assigned_group_name": group_name,
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def make_accident_place_from_traffic_type(tt_series: pd.Series) -> pd.Series:
    def mapper(x):
        try:
            t = int(x)
        except Exception:
            return 6  # Others

        if 0 <= t <= 20:
            return 0  # 직선도로
        if 21 <= t <= 59:
            return 1  # 사거리교차로(신호등 없음)
        if 60 <= t <= 105:
            return 2  # 사거리교차로(신호등 있음)
        if 106 <= t <= 122:
            return 3  # T자형 교차로
        if 123 <= t <= 127:
            return 4  # 차도 외/비도로 공간
        if 128 <= t <= 132:
            return 5  # 회전교차로
        return 6

    return tt_series.map(mapper).astype(int)


def get_accident_place_feature_groupings():
    # traffic_accident_type 기반 coarse grouping
    groups = {
        "추돌사고": expand_ranges((0, 2)),
        "합류/차로감소": expand_ranges(3),
        "개문/정차접촉": expand_ranges(4),
        "역주행/중앙선침범": expand_ranges(5),
        "동일폭 도로": expand_ranges(6, (21, 30), (106, 109)),
        "추월사고": expand_ranges((7, 10), 90),
        "진로/차로변경": expand_ranges(11, 12, 13, 52, 89),
        "안전지대 통과 사고": expand_ranges(14, 15),
        "정차후 출발 사고": expand_ranges(16, 58, 97),
        "긴급자동차 사고": expand_ranges((17, 20), 59, (103, 105)),
        "대로와 소로": expand_ranges((31, 43), (110, 116)),
        "한쪽방향 시설/조건": expand_ranges((44, 51), (80, 88), (117, 122)),
        "나란히 통행 가능한 차로폭": expand_ranges(53, 54),
        "좌/우회전": expand_ranges(55, 56, 57, 79),
        "상대 차량 진입": expand_ranges((60, 78)),
        "유턴 구역": expand_ranges((91, 96)),
        "노면 표시 위반 사고": expand_ranges((98, 102)),
        "차도 외/비도로 공간": expand_ranges((123, 127)),
        "회전차로": expand_ranges((128, 132)),
    }
    return groups


def get_vehicle_a_groupings():
    straight_base = expand_ranges(
        0, 3, 5, 6, 7, 8, 17, 19,
        (21, 26), (31, 36), 44, 45, 46, 47, 49, 51, 52, 58, 59,
        (60, 68), (71, 75), (80, 82), 89, 91, 92, 98, 99, 100, 103,
        (106, 108), (110, 113), 115, (117, 121), (123, 127)
    )
    straight_right = expand_ranges((27, 29), (37, 42), 48, 57, (83, 85), 93, 94, 102)
    straight_left = expand_ranges(30, 43, 50, 69, 70, (76, 79), (86, 88), 104, 109, 114, 116, 122)
    trailing = expand_ranges(1, 4, 11, 14, 15, 18, (53, 56))
    turning = expand_ranges((128, 132))
    uturn = expand_ranges(95, 96, 105)
    overtake = expand_ranges(9, 10, 90, 101)
    start_after_stop = expand_ranges(16, 97)
    lane_change = expand_ranges(12, 13, 20)
    parking_stop = expand_ranges(2)

    v1_groups = {
        "직진": sorted(set(straight_base + straight_right + straight_left)),
        "후행": trailing,
        "회전": turning,
        "유턴": uturn,
        "추월": overtake,
        "정차후출발": start_after_stop,
        "진로변경": lane_change,
        "주정차": parking_stop,
    }

    v2_groups = {
        "직진_선행or기타": straight_base,
        "직진_우회전결합": straight_right,
        "직진_좌회전결합": straight_left,
        "후행": trailing,
        "회전": turning,
        "유턴": uturn,
        "추월": overtake,
        "정차후출발": start_after_stop,
        "진로변경": lane_change,
        "주정차": parking_stop,
    }

    return v1_groups, v2_groups


def get_vehicle_b_groupings():
    straight_base = expand_ranges(
        0, 1, 3, (4, 6), 10, 13, 14, 15, 18,
        (21, 23), (27, 29), 31, 33, 34, (37, 42), 44, 48, 51, 59,
        (60, 63), 65, 69, 70, 76, 77, (79, 88),
        101, 103, 104, 105, 126
    )
    straight_right = expand_ranges(49, 52, 54, 56, 57, 78, 89, 106, 111, 115, 118, 121)
    straight_left = expand_ranges(
        (24, 26), 30, 32, 35, 36, 43, (45, 47), 50, 53, 55, 64,
        (66, 68), (71, 75), 90, (98, 100), (107, 110),
        (112, 114), 116, 117, 119, 120, 122
    )
    trailing = expand_ranges(2, 20, 102, 127)
    turning = expand_ranges((128, 132))
    uturn = expand_ranges((91, 96))
    overtake = expand_ranges((7, 9), 16, 19, 58, 97)
    lane_change = expand_ranges(11, 12)
    non_road_entry = expand_ranges(123, 124, 125)
    special_traffic = expand_ranges(17)

    v1_groups = {
        "직진": sorted(set(straight_base + straight_right + straight_left)),
        "후행": trailing,
        "회전": turning,
        "유턴": uturn,
        "추월": overtake,
        "진로변경": lane_change,
        "차도가아닌장소진입": non_road_entry,
        "특수통행": special_traffic,
    }

    v2_groups = {
        "직진_선행or기타": straight_base,
        "직진_우회전결합": straight_right,
        "직진_좌회전결합": straight_left,
        "후행": trailing,
        "회전": turning,
        "유턴": uturn,
        "추월": overtake,
        "진로변경": lane_change,
        "차도가아닌장소진입": non_road_entry,
        "특수통행": special_traffic,
    }

    return v1_groups, v2_groups


def validation_rows(
    feature_name: str,
    fine_series: pd.Series,
    coarse_series: pd.Series,
    fine_to_group: Dict[int, str],
    coarse_codebook: Dict[str, int],
    duplicates: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    used_codes = sorted(set(int(x) for x in fine_series.dropna().astype(int).tolist()))
    rows = []

    for code in used_codes:
        grp = fine_to_group.get(code, "Others")
        rows.append({
            "feature": feature_name,
            "fine_code": code,
            "coarse_code": coarse_codebook[grp],
            "coarse_name": grp,
            "is_others": grp == "Others",
        })

    map_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame([{
        "feature": feature_name,
        "n_rows": int(len(fine_series)),
        "used_fine_codes": int(len(used_codes)),
        "mapped_codes_not_others": int(sum(1 for c in used_codes if fine_to_group.get(c, "Others") != "Others")),
        "mapped_codes_to_others": int(sum(1 for c in used_codes if fine_to_group.get(c, "Others") == "Others")),
        "n_duplicate_conflicts_in_definition": int(len(duplicates)),
        "others_rows": int((coarse_series == coarse_codebook["Others"]).sum()),
    }])

    return map_df, summary_df


def main():
    args = parse_args()
    tag = experiment_tag(args.experiment, args.model, args.epoch)

    input_csv = Path(args.input_csv) if args.input_csv else BASE_FINE_OUTPUT_DIR / tag / "merged_analysis_table.csv"
    output_root = Path(args.output_dir) if args.output_dir else BASE_COARSE_OUTPUT_DIR / tag

    dictionaries_dir = output_root / "coarse_build" / "dictionaries"
    mappings_dir = output_root / "coarse_build" / "fine_to_coarse_mappings"
    validation_dir = output_root / "coarse_build" / "validation"

    for d in [output_root, dictionaries_dir, mappings_dir, validation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[1/4] Loading merged_analysis_table.csv")
    print(f"Input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded rows: {len(df)}")

    required = [
        "video_name", "cluster", "traffic_accident_type",
        "accident_place", "accident_place_feature",
        "vehicle_a_progress_info", "vehicle_b_progress_info",
        "accident_negligence_rateA", "accident_negligence_rateB",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("=" * 80)
    print("[2/4] Building coarse columns")

    # accident_place coarse
    accident_place_codebook = build_coarse_codebook([
        "직선도로",
        "사거리교차로(신호등없음)",
        "사거리교차로(신호등있음)",
        "T자형교차로",
        "차도외/비도로공간",
        "회전교차로",
    ])
    df["accident_place_coarse"] = make_accident_place_from_traffic_type(df["traffic_accident_type"])

    # accident_place_feature coarse (traffic_accident_type 기반)
    apf_group_to_codes = get_accident_place_feature_groupings()
    apf_code_to_group, apf_duplicates, _ = invert_group_dict(apf_group_to_codes)
    apf_codebook = build_coarse_codebook(list(apf_group_to_codes.keys()))
    df["accident_place_feature_coarse"] = apply_mapping(df["traffic_accident_type"], apf_code_to_group, apf_codebook)

    # vehicle A coarse
    va_v1_groups, va_v2_groups = get_vehicle_a_groupings()
    va_v1_code_to_group, va_v1_duplicates, _ = invert_group_dict(va_v1_groups)
    va_v2_code_to_group, va_v2_duplicates, _ = invert_group_dict(va_v2_groups)
    va_v1_codebook = build_coarse_codebook(list(va_v1_groups.keys()))
    va_v2_codebook = build_coarse_codebook(list(va_v2_groups.keys()))
    df["vehicle_a_progress_info_coarse_v1"] = apply_mapping(df["vehicle_a_progress_info"], va_v1_code_to_group, va_v1_codebook)
    df["vehicle_a_progress_info_coarse_v2"] = apply_mapping(df["vehicle_a_progress_info"], va_v2_code_to_group, va_v2_codebook)

    # vehicle B coarse
    vb_v1_groups, vb_v2_groups = get_vehicle_b_groupings()
    vb_v1_code_to_group, vb_v1_duplicates, _ = invert_group_dict(vb_v1_groups)
    vb_v2_code_to_group, vb_v2_duplicates, _ = invert_group_dict(vb_v2_groups)
    vb_v1_codebook = build_coarse_codebook(list(vb_v1_groups.keys()))
    vb_v2_codebook = build_coarse_codebook(list(vb_v2_groups.keys()))
    df["vehicle_b_progress_info_coarse_v1"] = apply_mapping(df["vehicle_b_progress_info"], vb_v1_code_to_group, vb_v1_codebook)
    df["vehicle_b_progress_info_coarse_v2"] = apply_mapping(df["vehicle_b_progress_info"], vb_v2_code_to_group, vb_v2_codebook)

    preview_cols = [
        "accident_place_coarse",
        "accident_place_feature_coarse",
        "vehicle_a_progress_info_coarse_v1",
        "vehicle_a_progress_info_coarse_v2",
        "vehicle_b_progress_info_coarse_v1",
        "vehicle_b_progress_info_coarse_v2",
    ]
    print("Added coarse columns:")
    print(df[preview_cols].head().to_string(index=False))

    print("=" * 80)
    print("[3/4] Saving dictionaries, mappings, and validation")

    # dictionaries
    save_dictionary_csv(dictionaries_dir / "accident_place_coarse_dictionary.csv", accident_place_codebook)
    save_dictionary_csv(dictionaries_dir / "accident_place_feature_coarse_dictionary.csv", apf_codebook)
    save_dictionary_csv(dictionaries_dir / "vehicle_a_progress_info_coarse_v1_dictionary.csv", va_v1_codebook)
    save_dictionary_csv(dictionaries_dir / "vehicle_a_progress_info_coarse_v2_dictionary.csv", va_v2_codebook)
    save_dictionary_csv(dictionaries_dir / "vehicle_b_progress_info_coarse_v1_dictionary.csv", vb_v1_codebook)
    save_dictionary_csv(dictionaries_dir / "vehicle_b_progress_info_coarse_v2_dictionary.csv", vb_v2_codebook)

    # mappings
    ap_rows = []
    for fine_code in sorted(set(df["traffic_accident_type"].dropna().astype(int).tolist())):
        coarse_code = int(make_accident_place_from_traffic_type(pd.Series([fine_code])).iloc[0])
        coarse_name = next(k for k, v in accident_place_codebook.items() if v == coarse_code)
        ap_rows.append({
            "fine_code": fine_code,
            "coarse_code": coarse_code,
            "coarse_name": coarse_name,
            "assigned_group_name": coarse_name,
        })
    pd.DataFrame(ap_rows).to_csv(
        mappings_dir / "traffic_accident_type_to_accident_place_coarse.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_fine_to_coarse_mapping(
        mappings_dir / "traffic_accident_type_to_accident_place_feature_coarse.csv",
        df["traffic_accident_type"],
        apf_code_to_group,
        apf_codebook,
    )
    save_fine_to_coarse_mapping(
        mappings_dir / "vehicle_a_progress_info_to_coarse_v1.csv",
        df["vehicle_a_progress_info"],
        va_v1_code_to_group,
        va_v1_codebook,
    )
    save_fine_to_coarse_mapping(
        mappings_dir / "vehicle_a_progress_info_to_coarse_v2.csv",
        df["vehicle_a_progress_info"],
        va_v2_code_to_group,
        va_v2_codebook,
    )
    save_fine_to_coarse_mapping(
        mappings_dir / "vehicle_b_progress_info_to_coarse_v1.csv",
        df["vehicle_b_progress_info"],
        vb_v1_code_to_group,
        vb_v1_codebook,
    )
    save_fine_to_coarse_mapping(
        mappings_dir / "vehicle_b_progress_info_to_coarse_v2.csv",
        df["vehicle_b_progress_info"],
        vb_v2_code_to_group,
        vb_v2_codebook,
    )

    # validation
    val_frames = []
    sum_frames = []

    used_tt = sorted(set(df["traffic_accident_type"].dropna().astype(int).tolist()))
    ap_map_rows = []
    for fine_code in used_tt:
        coarse_code = int(make_accident_place_from_traffic_type(pd.Series([fine_code])).iloc[0])
        coarse_name = next(k for k, v in accident_place_codebook.items() if v == coarse_code)
        ap_map_rows.append({
            "feature": "accident_place_coarse",
            "fine_code": fine_code,
            "coarse_code": coarse_code,
            "coarse_name": coarse_name,
            "is_others": coarse_name == "Others",
        })
    ap_map_df = pd.DataFrame(ap_map_rows)
    ap_sum_df = pd.DataFrame([{
        "feature": "accident_place_coarse",
        "n_rows": int(len(df)),
        "used_fine_codes": int(len(used_tt)),
        "mapped_codes_not_others": int((ap_map_df["is_others"] == False).sum()),
        "mapped_codes_to_others": int((ap_map_df["is_others"] == True).sum()),
        "n_duplicate_conflicts_in_definition": 0,
        "others_rows": int((df["accident_place_coarse"] == accident_place_codebook["Others"]).sum()),
    }])
    val_frames.append(ap_map_df)
    sum_frames.append(ap_sum_df)

    for feature_name, fine_series, coarse_series, code_to_group, codebook, duplicates in [
        ("accident_place_feature_coarse", df["traffic_accident_type"], df["accident_place_feature_coarse"], apf_code_to_group, apf_codebook, apf_duplicates),
        ("vehicle_a_progress_info_coarse_v1", df["vehicle_a_progress_info"], df["vehicle_a_progress_info_coarse_v1"], va_v1_code_to_group, va_v1_codebook, va_v1_duplicates),
        ("vehicle_a_progress_info_coarse_v2", df["vehicle_a_progress_info"], df["vehicle_a_progress_info_coarse_v2"], va_v2_code_to_group, va_v2_codebook, va_v2_duplicates),
        ("vehicle_b_progress_info_coarse_v1", df["vehicle_b_progress_info"], df["vehicle_b_progress_info_coarse_v1"], vb_v1_code_to_group, vb_v1_codebook, vb_v1_duplicates),
        ("vehicle_b_progress_info_coarse_v2", df["vehicle_b_progress_info"], df["vehicle_b_progress_info_coarse_v2"], vb_v2_code_to_group, vb_v2_codebook, vb_v2_duplicates),
    ]:
        map_df, sum_df = validation_rows(feature_name, fine_series, coarse_series, code_to_group, codebook, duplicates)
        val_frames.append(map_df)
        sum_frames.append(sum_df)

    pd.concat(val_frames, ignore_index=True).to_csv(
        validation_dir / "fine_to_coarse_validation_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(sum_frames, ignore_index=True).to_csv(
        validation_dir / "coarse_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dup_rows = []
    for feature_name, duplicates in [
        ("accident_place_feature_coarse", apf_duplicates),
        ("vehicle_a_progress_info_coarse_v1", va_v1_duplicates),
        ("vehicle_a_progress_info_coarse_v2", va_v2_duplicates),
        ("vehicle_b_progress_info_coarse_v1", vb_v1_duplicates),
        ("vehicle_b_progress_info_coarse_v2", vb_v2_duplicates),
    ]:
        for item in duplicates:
            dup_rows.append({"feature": feature_name, **item})
    pd.DataFrame(dup_rows).to_csv(
        validation_dir / "duplicate_code_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(pd.concat(sum_frames, ignore_index=True).to_string(index=False))

    print("=" * 80)
    print("[4/4] Saving merged_analysis_table_coarse.csv")
    coarse_csv = output_root / "merged_analysis_table_coarse.csv"
    df.to_csv(coarse_csv, index=False, encoding="utf-8-sig")
    print(f"Saved coarse table: {coarse_csv}")
    print(f"Saved outputs under: {output_root}")
    print("Done.")


if __name__ == "__main__":
    main()
