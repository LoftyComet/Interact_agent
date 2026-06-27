"""
为 term_inventory.json 树结构添加 Excel 中的子分组层级。

Excel 中的子分组（【单点点击】、【点击类冲突调和】等）在 JSON 中缺失，
本脚本将这些中间节点插入到树中。
"""

import json

# 子分组定义：每个 section 下的 subgroup 及其包含的 leaf labels
# 格式: { section_label: [ (subgroup_label, [leaf_label, ...]), ... ] }

SUBGROUPS = {
    "点击类": [
        ("单点点击", ["开关", "单击", "按下", "持续触发", "长按", "双击"]),
        ("多点点击", ["多点开关", "多点有序点击", "多点同时点击"]),
    ],
    "位移类": [
        ("单点位移", ["拖拽", "甩动", "翻动", "滑动切换", "越界切换", "晃动·震动"]),
        ("多点位移", ["捏合缩放", "环绕旋转", "整体移动"]),
    ],
    "冲突调和": [
        ("点击类冲突调和", ["快击·快抬", "点击缓冲", "缓冲连发"]),
        ("位移类冲突调和", ["边缘滑入", "滑入停留", "方向解耦", "轻拨"]),
        ("其他冲突调和", ["点拖互斥", "捏合解耦"]),
    ],
    "实体": [
        ("点触实体", ["按钮", "拨钮/杆", "滚轮", "指摇杆", "轨迹球", "指点杆", "触控面"]),
        ("握持实体", ["插头-插座", "旋钮", "旋转手柄", "钳形手柄", "摇柄", "推拉杆", "手摇杆", "鼠标/自由手柄", "圆环", "绳", "笔"]),
        ("其他实体", ["踏板", "履带", "门板窗", "座椅", "载具", "柔体"]),
    ],
    "姿态": [
        ("头", ["眼睛", "嘴巴", "舌头", "面部", "头部"]),
        ("手势", ["指尖", "手", "手臂"]),
        ("其他", ["腿脚", "胸腹", "全身"]),
    ],
}


def apply_subgroups(node, section_label, subgroup_defs):
    """
    在 section 节点的 children 中插入 subgroup 中间节点。
    subgroup_defs: [(subgroup_label, [leaf_label, ...]), ...]
    """
    children = node.get('children', [])
    if not children:
        return

    # 构建 label → child 索引
    child_by_label = {}
    for c in children:
        child_by_label[c['label']] = c

    # 构建新的 children 列表
    new_children = []
    used_labels = set()

    for sub_label, leaf_labels in subgroup_defs:
        subgroup = {
            "label": sub_label,
            "children": []
        }
        for leaf_label in leaf_labels:
            if leaf_label in child_by_label:
                subgroup["children"].append(child_by_label[leaf_label])
                used_labels.add(leaf_label)
        if subgroup["children"]:
            new_children.append(subgroup)

    # 添加未归入任何子分组的叶子节点（如含义识别等）
    for c in children:
        if c['label'] not in used_labels:
            new_children.append(c)

    node['children'] = new_children


def main():
    with open('data/term_inventory.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    root = data['root']

    # 定位各 section 节点
    for child in root['children']:
        if child['label'] == '交互逻辑':
            for sub in child['children']:
                sec_label = sub['label']
                if sec_label in SUBGROUPS:
                    apply_subgroups(sub, sec_label, SUBGROUPS[sec_label])
                    # 统计
                    sg_count = len(sub['children'])
                    leaf_count = sum(len(sg.get('children', [])) for sg in sub['children'])
                    print(f"  {sec_label}: {sg_count} subgroups, {leaf_count} leaves")

        elif child['label'] == '控件形态':
            for sub in child['children']:
                sec_label = sub['label']
                if sec_label in SUBGROUPS:
                    apply_subgroups(sub, sec_label, SUBGROUPS[sec_label])
                    sg_count = len(sub['children'])
                    leaf_count = sum(len(sg.get('children', [])) for sg in sub['children'])
                    print(f"  {sec_label}: {sg_count} subgroups, {leaf_count} leaves")

    # 保存
    with open('data/term_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n✅ Subgroups added to JSON tree")


if __name__ == '__main__':
    main()
