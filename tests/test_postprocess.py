import pytest

from sav_parsers.fpb_mod1 import _REGION_PAIRS
from sav_parsers.postprocess import SlotOffset, pair_region_bboxes, slot_from_anchor
from sav_parsers.types import BBox, ParsedField


def assert_vertices(vertices, expected):
  for actual, wanted in zip(vertices, expected, strict=True):
    assert actual == pytest.approx(wanted, abs=1e-5)


def test_slot_from_anchor_licenca_calibration():
  vertices = slot_from_anchor(
    [(0.079714455, 0.15138772), (0.1522903, 0.16021867)],
    SlotOffset(right=1.49, width=1.12, height=1.93),
  )

  assert_vertices(vertices, [
    (0.18350, 0.14728),
    (0.26478, 0.14728),
    (0.26478, 0.16433),
    (0.18350, 0.16433),
  ])


def test_slot_from_anchor_carimbo_calibration():
  vertices = slot_from_anchor(
    [(0.6912552, 0.7401177), (0.8334325, 0.7493692)],
    SlotOffset(up=2.75, width=5.5, height=5.5),
  )

  assert_vertices(vertices, [
    (0.37136, 0.69386),
    (1.15333, 0.69386),
    (1.15333, 0.74474),
    (0.37136, 0.74474),
  ])


def test_slot_from_anchor_identity():
  anchor = [(0.2, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7)]

  assert_vertices(slot_from_anchor(anchor, SlotOffset()), anchor)


def test_pair_pops_region_and_synthesizes_presence_from_value():
  fields = {
    "licenca_fpb_region": ParsedField(
      value=None,
      confidence=0.61,
      bbox=BBox(
        page=2,
        vertices=[(0.079714455, 0.15138772), (0.1522903, 0.16021867)],
      ),
    ),
    "licenca_fpb": ParsedField(value="12345", confidence=0.92),
  }

  pair_region_bboxes(fields, _REGION_PAIRS)

  assert "licenca_fpb_region" not in fields
  presence = fields["licenca_fpb_presente"]
  assert presence.value is True
  assert presence.confidence == 0.61
  assert presence.bbox is not None
  assert presence.bbox.page == 2
  assert_vertices(presence.bbox.vertices, [
    (0.18350, 0.14728),
    (0.26478, 0.14728),
    (0.26478, 0.16433),
    (0.18350, 0.16433),
  ])


@pytest.mark.parametrize("value, expected", [("12345", True), (None, False)])
def test_licenca_presence_follows_value(value, expected):
  fields = {
    "licenca_fpb_region": ParsedField(
      value=None,
      confidence=0.61,
      bbox=BBox(page=0, vertices=[(0.1, 0.2), (0.2, 0.3)]),
    ),
    "licenca_fpb": ParsedField(value=value, confidence=0.92),
    "licenca_fpb_presente": ParsedField(value=None, confidence=0.0),
  }

  pair_region_bboxes(fields, _REGION_PAIRS)

  assert fields["licenca_fpb_presente"].value is expected


def test_existing_carimbo_presence_value_wins():
  fields = {
    "carimbo_clube_region": ParsedField(
      value=None,
      confidence=0.73,
      bbox=BBox(page=1, vertices=[(0.6912552, 0.7401177), (0.8334325, 0.7493692)]),
    ),
    "carimbo_clube_presente": ParsedField(value=True, confidence=0.98),
  }

  pair_region_bboxes(fields, _REGION_PAIRS)

  presence = fields["carimbo_clube_presente"]
  assert presence.value is True
  assert presence.confidence == 0.98
  assert presence.bbox is not None
  assert_vertices(presence.bbox.vertices, [
    (0.37136, 0.69386),
    (1.15333, 0.69386),
    (1.15333, 0.74474),
    (0.37136, 0.74474),
  ])


def test_absent_value_entity_yields_false_presence():
  fields = {
    "licenca_fpb_region": ParsedField(
      value=None,
      confidence=0.61,
      bbox=BBox(page=0, vertices=[(0.1, 0.2), (0.2, 0.3)]),
    ),
  }

  pair_region_bboxes(fields, _REGION_PAIRS)

  presence = fields["licenca_fpb_presente"]
  assert presence.value is False
  assert presence.bbox is not None


def test_missing_region_leaves_presence_untouched():
  presence = ParsedField(
    value=None,
    confidence=0.42,
    bbox=BBox(page=4, vertices=[(0.1, 0.2), (0.3, 0.4)]),
  )
  fields = {"carimbo_clube_presente": presence}

  pair_region_bboxes(fields, _REGION_PAIRS)

  assert fields["carimbo_clube_presente"] is presence
  assert presence.value is None
  assert presence.confidence == 0.42
  assert presence.bbox == BBox(page=4, vertices=[(0.1, 0.2), (0.3, 0.4)])
