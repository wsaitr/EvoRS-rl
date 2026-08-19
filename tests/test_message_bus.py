from rs_hmcomm.core import AgentMessage, MessageBus, MessageModality, NodeLevel

def test_message_stats():
    bus = MessageBus()
    bus.send(AgentMessage(
        sender="global",
        receiver="local",
        spatial_level=NodeLevel.REGION,
        modalities={MessageModality.TEXT, MessageModality.STRUCT},
        text="inspect region",
        node_ids=["r1"],
        bbox_refs=[(0,0,1,1)],
    ))
    s = bus.stats()
    assert s["messages"] == 1
    assert s["struct_nodes"] == 1
    assert s["bbox_refs"] == 1
