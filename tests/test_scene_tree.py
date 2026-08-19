from rs_hmcomm.core import SceneTree, SceneNode, NodeLevel, NodeStatus

def test_tree_add_query_verify():
    tree = SceneTree("airport_candidate")
    region = SceneNode.make(NodeLevel.REGION, "apron", bbox=(0.1,0.1,0.8,0.8), confidence=0.7)
    tree.add_node(region)
    group = SceneNode.make(NodeLevel.GROUP, "aircraft_group", parent_id=region.id, confidence=0.8)
    tree.add_node(group)
    assert tree.children(region.id)[0].id == group.id
    assert tree.query_by_semantic("aircraft")[0].id == group.id
    tree.verify(group.id, 0.95)
    assert tree.get_node(group.id).status == NodeStatus.VERIFIED
    assert tree.get_node(group.id).confidence == 0.95
