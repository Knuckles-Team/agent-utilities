"""Shared BPMN 2.0 fixture for the step-level lift (KG-2.53) and the
ontology→workflow bridge compiler (ORCH-1.41) tests.

Topology: start → review (userTask) → decide (exclusiveGateway)
  → [approved] archive (serviceTask) → end
  → [else]     rework  (task) → notify (event, collapsed) → archive
"""

from __future__ import annotations

BPMN_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   id="defs1" targetNamespace="http://example.test/bpmn">
  <bpmn2:process id="invoice" name="Invoice Receipt" isExecutable="true">
    <bpmn2:startEvent id="start1"/>
    <bpmn2:userTask id="review" name="Review Invoice"/>
    <bpmn2:exclusiveGateway id="decide" name="Approved?"/>
    <bpmn2:serviceTask id="archive" name="Archive Invoice"/>
    <bpmn2:task id="rework" name="Request Rework"/>
    <bpmn2:intermediateThrowEvent id="notify1"/>
    <bpmn2:endEvent id="end1"/>
    <bpmn2:sequenceFlow id="f1" sourceRef="start1" targetRef="review"/>
    <bpmn2:sequenceFlow id="f2" sourceRef="review" targetRef="decide"/>
    <bpmn2:sequenceFlow id="f3" sourceRef="decide" targetRef="archive">
      <bpmn2:conditionExpression>${approved == true}</bpmn2:conditionExpression>
    </bpmn2:sequenceFlow>
    <bpmn2:sequenceFlow id="f4" sourceRef="decide" targetRef="rework">
      <bpmn2:conditionExpression>${approved == false}</bpmn2:conditionExpression>
    </bpmn2:sequenceFlow>
    <bpmn2:sequenceFlow id="f5" sourceRef="rework" targetRef="notify1"/>
    <bpmn2:sequenceFlow id="f6" sourceRef="notify1" targetRef="archive"/>
    <bpmn2:sequenceFlow id="f7" sourceRef="archive" targetRef="end1"/>
  </bpmn2:process>
</bpmn2:definitions>
"""


class XmlCapableClient:
    """Camunda client with the optional camunda_process action=xml capability."""

    def list_process_definitions(self):
        return [{"id": "invoice:1:abc", "key": "invoice", "name": "Invoice Receipt"}]

    def get_process_definition_xml(self, id=None, key=None):
        # Camunda 7 REST envelope shape.
        return {"id": id, "bpmn20Xml": BPMN_FIXTURE}
