from __future__ import print_function

import collections
import configparser
import os
import glob
import json
import math
import argparse
import itertools

import cv2
import numpy as np
import PyQt5.QtWidgets as qt

from extract import extract, _size
from temporal_lists import TemporalList, RangedTemporalList, SequentialTemporalList


def _format_time(seconds):
    s = math.floor(seconds)
    ms = math.floor((seconds - s) * 1000)
    m = s // 60
    h = m // 60
    s -= m * 60 + h * 3600
    m -= h * 60
    if h > 0:
        return "%02d:%02d:%02d:%03d" % (h, m, s, ms)
    return "%02d:%02d:%03d" % (m, s, ms)


def _scan_frames(video):
    frames_dir = os.path.join(os.path.dirname(os.path.abspath(video)), os.path.basename(os.path.splitext(video)[0]))
    if not os.path.exists(frames_dir):
        return None, None, True
    
    with open(os.path.join(frames_dir, "metadata.json"), "r") as r:
        metadata = json.load(r, object_hook=lambda d: collections.namedtuple("Metadata", d.keys())(*d.values()))
    
    frame_images = [os.path.join(frames_dir, "%d.jpg" % (i * metadata.frame_interval,)) for i in range(int(metadata.frame_count // metadata.frame_interval))]
    
    return metadata, frame_images, len(glob.glob(os.path.join(frames_dir, "*.jpg"))) < len(frame_images)


def _draw_text(img, text, org, text_font, halign="left", valign="bottom", padding=4, scale=1, thickness=1, color=(0, 255, 0), background_color=(0, 0, 0)):
    if text is None or len(text) == 0:
        return img
    
    org_x, org_rel_x = org[0], 0.
    if type(org[0]) == complex:
        org_rel_x = org[0].imag
        org_x = org[0].real
    
    org_y, org_rel_y = org[1], 0.
    if type(org[1]) == complex:
        org_rel_y = org[1].imag
        org_y = org[1].real

    size, _ = cv2.getTextSize(text, text_font, scale, thickness)
    
    if halign == "right":
        tl_x = org_x - size[0] - padding * 2
        br_x = org_x
        orig_x = org_x - size[0] - padding
    elif halign == "center":
        tl_x = org_x - size[0] / 2 - padding
        br_x = org_x + size[0] / 2 + padding
        orig_x = org_x - size[0] / 2
    else:
        tl_x = org_x
        br_x = org_x + size[0] + padding * 2
        orig_x = org_x + padding
        
    if valign == "top":
        tl_y = org_y
        br_y = org_y + size[1] + padding * 2
        orig_y = org_y + size[1] + padding
    elif valign == "center":
        tl_y = org_y - size[1] / 2 - padding
        br_y = org_y + size[1] / 2 + padding
        orig_y = org_y + size[1] / 2 + padding
    else:
        tl_y = org_y - size[1] - padding * 2
        br_y = org_y
        orig_y = org_y - padding
    
    rel_offset_x = org_rel_x * (size[0] + padding * 2)
    rel_offset_y = org_rel_y * (size[1] + padding * 2)
    
    if background_color is not None:
        img = cv2.rectangle(img, (int(tl_x + rel_offset_x), int(tl_y + rel_offset_y)), (int(br_x + rel_offset_x), int(br_y + rel_offset_y)), background_color, -1)
    
    return cv2.putText(img, text, (int(orig_x + rel_offset_x), int(orig_y + rel_offset_y)), text_font, scale, color, thickness)


def _update_kill_item_player_name(kill_item, old_name, new_name):
    if kill_item["name"] == old_name:
        kill_item["name"] = new_name


def _update_kill_player_name(kill, old_name, new_name):
    if kill["killer"] is not None:
        _update_kill_item_player_name(kill["killer"], old_name, new_name)
    _update_kill_item_player_name(kill["killee"], old_name, new_name)
    for assist in kill["assists"]:
        _update_kill_item_player_name(assist, old_name, new_name)


def _update_kill_item_player_hero(kill_item, player_name, new_hero):
    if kill_item["name"] == player_name:
        kill_item["hero"] = new_hero

def _update_kill_player_hero(kill, player_name, new_hero):
    if kill["killer"] is not None:
        _update_kill_item_player_hero(kill["killer"], player_name, new_hero)
    _update_kill_item_player_hero(kill["killee"], player_name, new_hero)
    for assist in kill["assists"]:
        _update_kill_item_player_hero(assist, player_name, new_hero)


class _HeroAbility:
    def __init__(self, ability_def):
        self.name = ability_def["name"]
        self.isUltimate = ability_def["ultimate"]


class _Hero:
    def __init__(self, hero_def):
        self.name = hero_def["name"]
        self.abilities = [_HeroAbility(ability_def) for ability_def in hero_def["abilities"]]
        try:
            self.alts = hero_def["alts"]
        except KeyError:
            self.alts = []


def _read_heroes_def_file(file):
    with open(file, "r") as r:
        return [_Hero(hero_def) for hero_def in json.load(r)["heroes"]]


def _format_kill_hero(kill_hero):
    return kill_hero["name"] + ":" + kill_hero["hero"]


def _format_kill(kill):
    kill_str = "-> " + _format_kill_hero(kill["killee"])
    if kill["ability"] is not None:
        kill_str = "-[" + kill["ability"] + "]" + kill_str
    
    try:
        if kill["critical"]:
            kill_str = "*" + kill_str
    except KeyError:
        pass
    
    if kill["killer"] is not None:
        for assist in kill["assists"]:
            kill_str = "& " + _format_kill_hero(assist) + " " + kill_str
        
        kill_str = _format_kill_hero(kill["killer"]) + " " + kill_str
    
    return kill_str

class _Descriptor:
    def __init__(self, description_file):
        self.__description_file = description_file
        self.set_updated()
        try:
            with open(description_file, "r") as r:
                self.__description = json.load(r)
            
            try:
                self.__description["matches"] = RangedTemporalList(self.__description["matches"])
            except KeyError:
                self.__description["matches"] = RangedTemporalList()
            
            for match in self.__description["matches"]:
                for player in match["players"]:
                    try:
                        player["heroes"] = SequentialTemporalList(player["heroes"])
                    except KeyError:
                        player["heroes"] = SequentialTemporalList()
                
                try:
                    match["kills"] = TemporalList(match["kills"])
                except KeyError:
                    match["kills"] = TemporalList()
            
        except FileNotFoundError:
            self.__description = {
                "matches": RangedTemporalList(),
            }
    
    def save(self):
        with open(self.__description_file, "w") as w:
            json.dump(self.__description, w, separators=(',', ':'))
    
    def set_updated(self):
        self.__updated = True
    
    def updated(self):
        if self.__updated:
            self.__updated = False
            return True
        return False
    
    def matches(self):
        return self.__description["matches"]
    
    def current_match(self, time):
        current_match = self.matches().current(time)
        if current_match is None:
            raise ValueError("There is not match at the current time")
        
        return current_match
    
    def add_match(self, start_time, end_time, map="", game_mode=""):
        matches = self.matches()
        if matches.current(start_time) is not None or matches.current(end_time) is not None:
            raise ValueError("matches may not overlap")
        
        prev_match = matches.prev(start_time)
        
        matches.append({
            "map": map,
            "game_mode": game_mode,
            "start_time": start_time,
            "end_time": end_time,
            "players": [
                {
                    "name": "" if prev_match is None else prev_match["players"][i]["name"],
                    "heroes": SequentialTemporalList(),
                } for i in range(12)
            ],
            "kills": TemporalList(),
        })
        
        matches.sort(key=lambda m: m["start_time"])
        
        for i, match in enumerate(matches):
            match["name"] = "Match %d" % (i + 1)
        
        self.set_updated()
    
    def update_match_end(self, time):
        match = self.matches().current(time)
        if match == None:
            match = self.matches().prev(time)
        match["end_time"] = time
        self.set_updated()
    
    def update_match_start(self, time):
        match = self.matches().current(time)
        if match == None:
            match = self.matches().next(time)
        match["start_time"] = time
        self.set_updated()
    
    def remove_match(self, time):
        self.matches().remove(time)
        self.set_updated()
    
    def player(self, time, position_or_name, current_match=None):
        if current_match is None:
            current_match = self.current_match(time)
        
        if type(position_or_name) != str:
            return current_match["players"][position_or_name]
        
        for player in current_match["players"]:
            if player["name"] == position_or_name:
                return player
        
        return None
    
    def player_heroes(self, time, position_or_name, current_match=None):
        return self.player(time, position_or_name, current_match=current_match)["heroes"]
    
    def update_player_name(self, time, position_or_name, name):
        current_match = self.current_match(time)
        player = self.player(time, position_or_name, current_match)
        for kill in current_match["kills"]:
            _update_kill_player_name(kill, player["name"], name)
                
        player["name"] = name
        self.set_updated()
    
    def update_player_hero(self, time, position_or_name, hero):
        current_match = self.current_match(time)
        
        player = self.player(time, position_or_name, current_match=current_match)
        player_heroes = player["heroes"]
        
        prev_hero = player_heroes.prev(time)
        current_hero = player_heroes.current(time)
        next_hero = player_heroes.next(time)
        
        if current_hero is not None and current_hero["start_time"] == time:
            if prev_hero is not None and prev_hero["name"] == hero:
                player_heroes.remove(time)
            else:
                current_hero["name"] = hero
        elif current_hero is None or current_hero != hero:
            player_heroes.append({
                "name": hero,
                "start_time": time,
            })
        
        if next_hero is not None and next_hero["name"] == hero:
            player_heroes.remove(next_hero["start_time"])
        
        current_hero = player_heroes.current(time)
        next_hero = player_heroes.next(time)
        
        for kill in current_match["kills"]:
            if next_hero is not None and kill["start_time"] >= next_hero["start_time"]:
                break
            
            if current_hero["start_time"] > kill["start_time"]:
                continue
            
            _update_kill_player_hero(kill, player["name"], hero)
        
        self.set_updated()
    
    def remove_player_hero(self, time, position_or_name):
        self.player_heroes(time, position_or_name).remove(time)
        self.set_updated()
    
    def kills(self, time, current_match=None):
        if current_match is None:
            current_match = self.current_match(time)
        
        return current_match["kills"]
    
    def __kill_hero(self, time, position_or_name, current_match=current_match):
        hero = None
        if type(position_or_name) == tuple:
            hero = position_or_name[1]
            position_or_name = position_or_name[0]
        
        if hero is None or type(position_or_name) != str:
            player = self.player(time, position_or_name, current_match)
            if hero is None:
                hero = player["heroes"].current(time)["name"]
            player = player["name"]
        else:
            player = position_or_name
        
        return {
            "name": player,
            "hero": hero,
        }
    
    def add_kill(self, time, killee_position_or_name, killer_position_or_name=None, assist_positions_or_names=[], ability=None, critical=False):
        current_match = self.current_match(time)
        self.kills(time, current_match=current_match).append({
            "start_time": time,
            "killer": None if killer_position_or_name is None else self.__kill_hero(time, killer_position_or_name, current_match),
            "assists": [
                self.__kill_hero(time, assist_position_or_name, current_match) for assist_position_or_name in assist_positions_or_names
            ],
            "killee": self.__kill_hero(time, killee_position_or_name, current_match),
            "ability": ability,
            "critical": critical,
        })
        self.set_updated()
    
    def update_kill(self, time, kill_index, killee_position_or_name, killer_position_or_name=None, assist_positions_or_names=[], ability=None, critical=False, current_match=None):
        if current_match is None:
            current_match = self.current_match(time)
        
        current_match["kills"][kill_index]["killer"] = None if killer_position_or_name is None else self.__kill_hero(time, killer_position_or_name, current_match)
        current_match["kills"][kill_index]["assists"] = [
                self.__kill_hero(time, assist_position_or_name, current_match) for assist_position_or_name in assist_positions_or_names
            ]
        current_match["kills"][kill_index]["killee"] = self.__kill_hero(time, killee_position_or_name, current_match)
        current_match["kills"][kill_index]["ability"] = ability
        current_match["kills"][kill_index]["critical"] = critical
        
        self.set_updated()
    
    def instantaneous_labels(self, time):
        current_match = self.matches().current(time)
        if current_match is None:
            return {
                "match": "",
                "player_names": [""] * 12,
                "player_heroes": [""] * 12,
                "kills": [],
            }
        
        return {
            "match": current_match["name"],
            "player_names": [
                current_match["players"][pos]["name"] for pos in range(12)
            ],
            "player_heroes": [
                ("" if player_hero is None else player_hero["name"]) for player_hero in [
                    current_match["players"][pos]["heroes"].current(time) for pos in range(12)
                ]
            ],
            "kills": [
                _format_kill(kill) for kill in reversed(current_match["kills"].prev(time, 6))
            ]
        }


class _VisualizerConfig:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)
        
        self.heroes = _read_heroes_def_file(config["General"]["HeroesJSON"] or "heroes.json")
        
        self.player_hero_points = tuple([
            _parse_point(config["Positions"][key]) for key in [("PlayerHero%d" % (i + 1)) for i in range(12)]
        ])
        
        self.player_hero_size = _parse_point(config["Sizes"]["PlayerHero"])
        
        self.kill_feed_pos = _parse_point(config["Positions"]["KillFeed"])


class _Visualizer:
    __name__ = "Visualizer"
    
    __STATE__MESSAGE = "toast_message"
    __STATE__OVERLAY = "hide_overlay"
    __STATE__CURSOR = "current_position"
    __STATE__EXTRACTION_PROGRESS = "extraction_progress"
    
    def __init__(self, config, video, descriptor, window_name="video", text_font=cv2.FONT_HERSHEY_PLAIN, draw_boxes=False):
        self.__heroes = config.heroes
        self.__descriptor = descriptor
        self.__window_name = window_name
        self.__text_font = text_font
        self.__draw_boxes = draw_boxes
        self.__kill_extraction = None
        self.__extraction_progress = None
        self.__render_cache = None
        self.__hide_overlay = False
        self.__message = None
        self.__focus = True
        self.__cursor = 0
        self.__player_hero_points = config.player_hero_points
        self.__player_hero_size = config.player_hero_size
        self.__kill_feed_pos = config.kill_feed_pos
        
        self.__update_state()
        
        self.__metadata, self.__frames, should_extract = _scan_frames(video)
        if should_extract:
            def progress(a, b):
                if a == b:
                    self.__extraction_progress = None
                    return
                
                self.__extraction_progress = (a, b)
            
            self.__metadata, self.__frames, self.__kill_extraction = extract(video, async=True, progress=progress, force=True)
            self.__extraction_progress = (0, len(self.__frames))
        
        if len(self.__frames) == 0:
            raise ValueError("no frames found")
    
    def __enter__(self):
        frames_per_second = math.ceil(self.__metadata.frame_rate / self.__metadata.frame_interval)
        
        small_frame_skip = 1
        medium_frame_skip = frames_per_second
        big_frame_skip = frames_per_second * 10
        
        def handle_mouse(event, x, y, flags, param):
            if not self.__focus:
                return
            
            if event == cv2.EVENT_MOUSEWHEEL:
                if flags & cv2.EVENT_FLAG_SHIFTKEY == cv2.EVENT_FLAG_SHIFTKEY:
                    frame_skip = big_frame_skip
                elif flags & cv2.EVENT_FLAG_CTRLKEY == cv2.EVENT_FLAG_CTRLKEY:
                    frame_skip = small_frame_skip
                else:
                    frame_skip = medium_frame_skip
            
                self.__cursor += frame_skip * (1 if flags < 0 else -1)
                
                if self.__cursor < 0:
                    self.__cursor = len(self.__frames) + self.__cursor
                elif self.__cursor >= len(self.__frames):
                    self.__cursor = self.__cursor - len(self.__frames)
                
                self.__message = None
            
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                self.__handle_double_click(x, y, flags)
        
        cv2.namedWindow(self.__window_name)
        cv2.setMouseCallback(self.__window_name, handle_mouse)
        
        self.__qt_app = qt.QApplication([])
        
        return self
    
    def __exit__(self, *args):
        if self.__kill_extraction is not None:
            self.__kill_extraction.set()
        
        cv2.destroyWindow(self.__window_name)
    
    def __lookup_hero(self, hero_name):
        for hero in self.__heroes:
            if hero.name == hero_name:
                return hero
        return None
    
    def __create_window(self, title):
        widget = qt.QWidget()
        widget.setWindowTitle(title)
        
        def focus_changed(_, new):
            if new is not None:
                return
            
            widget.close()
            self.__qt_app.focusChanged.disconnect(focus_changed)
            self.__focus = True
        
        self.__qt_app.focusChanged.connect(focus_changed)
        
        return widget
    
    def __open_match_buttons(self):
        widget = self.__create_window("Edit Matches")
        
        grid = qt.QGridLayout(widget)
        
        current_time = self.__current_time()
        
        matches = self.__descriptor.matches()
        prev_match = matches.prev(current_time)
        current_match = matches.current(current_time)
        next_match = matches.next(current_time)
        
        create_match_btn = qt.QPushButton("New Match")
        if current_match is None:
            def match_new():
                end_time = self.__total_time()
                
                if next_match is not None:
                    end_time = next_match["start_time"] - (self.__metadata.frame_interval / self.__metadata.frame_rate)
                    
                self.__descriptor.add_match(current_time, end_time)
                widget.close()
            create_match_btn.clicked.connect(match_new)
        else:
            create_match_btn.setEnabled(False)
        grid.addWidget(create_match_btn, 1, 1, 1, 1)
        
        delete_match_btn = qt.QPushButton("Delete Match")
        if current_match is not None:
            def match_remove():
                self.__descriptor.remove_match(current_time)
                widget.close()
            delete_match_btn.clicked.connect(match_remove)
        else:
            delete_match_btn.setEnabled(False)
        grid.addWidget(delete_match_btn, 1, 2, 1, 1)
        
        set_start_btn = qt.QPushButton("Update Start")
        if next_match is None and current_match is None:
            set_start_btn.setEnabled(False)
        else:
            def match_start():
                self.__descriptor.update_match_start(current_time)
                widget.close()
            set_start_btn.clicked.connect(match_start)
        grid.addWidget(set_start_btn, 2, 1, 1, 1)
        
        set_end_btn = qt.QPushButton("Update End")
        if prev_match is None and current_match is None:
            set_end_btn.setEnabled(False)
        else:
            def match_end():
                self.__descriptor.update_match_end(current_time)
                widget.close()
            set_end_btn.clicked.connect(match_end)
        grid.addWidget(set_end_btn, 2, 2, 1, 1)
        
        jump_prev_btn = qt.QPushButton("Jump Prev")
        if prev_match is None:
            jump_prev_btn.setEnabled(False)
        else:
            def jump_prev():
                self.__cursor = int(prev_match["start_time"] * self.__metadata.frame_rate / self.__metadata.frame_interval)
                widget.close()
            jump_prev_btn.clicked.connect(jump_prev)
        grid.addWidget(jump_prev_btn, 3, 1, 1, 1)
        
        jump_next_btn = qt.QPushButton("Jump Next")
        if next_match is None:
            jump_next_btn.setEnabled(False)
        else:
            def jump_next():
                self.__cursor = int((next_match["start_time"] * self.__metadata.frame_rate + 1.)/ self.__metadata.frame_interval)
                widget.close()
            jump_next_btn.clicked.connect(jump_next)
        grid.addWidget(jump_next_btn, 3, 2, 1, 1)
        
        self.__focus = False
        widget.show()
    
    def __show_warning(self, message, title="Warning"):
        qt.QMessageBox.warning(None, title, message, qt.QMessageBox.Ok)
    
    def __edit_player_hero(self, position):
        current_time = self.__current_time()
        try:
            current_match = self.__descriptor.current_match(current_time)
        except ValueError as e:
            self.__show_warning(str(e))
            return
        
        widget = self.__create_window("Edit Player %d" % position)
        
        player = self.__descriptor.player(current_time, position)
        player_hero = player["heroes"].current(current_time)
        
        grid = qt.QGridLayout(widget)
        
        name_edit = qt.QLineEdit(player["name"])
        grid.addWidget(name_edit, 1, 1, 1, 3)
        
        combo_box = qt.QComboBox()
        combo_box.addItem("")
        for hero in self.__heroes:
            combo_box.addItem(hero.name)
        
        grid.addWidget(combo_box, 2, 1, 1, 3)
        
        def update_player_hero():
            if player_hero is None or combo_box.currentText() != player_hero["name"]:
                self.__descriptor.update_player_hero(current_time, position, combo_box.currentText())
            
            if name_edit.text() != player["name"]:
                self.__descriptor.update_player_name(current_time, position, name_edit.text())
            
            widget.close()
        
        ok_btn = qt.QPushButton("Ok")
        ok_btn.clicked.connect(update_player_hero)
        grid.addWidget(ok_btn, 3, 2, 1, 1)
        
        def selection_changed(hero):
            if hero:
                ok_btn.setEnabled(True)
            else:
                ok_btn.setEnabled(False)
            
        combo_box.currentTextChanged.connect(selection_changed)
        
        if player_hero is not None:
            combo_box.setCurrentText(player_hero["name"])
        else:
            ok_btn.setEnabled(False)
        
        self.__focus = False
        widget.show()
    
    def __open_add_kill_window(self, edit=None):
        current_time = self.__current_time()
        try:
            current_match = self.__descriptor.current_match(current_time)
        except ValueError as e:
            self.__show_warning(str(e))
            return
        
        current_player_heroes = [
            (player["name"] + " - " + player["heroes"].current(current_time)["name"]) for player in current_match["players"]
        ]
        
        widget = self.__create_window("Add Kill" if edit is None else "Save Kill")
        
        grid = qt.QGridLayout(widget)
        
        
        killer_combo_box = qt.QComboBox()
        killer_combo_box.addItem("")
        killer_combo_box.addItems(current_player_heroes)
        grid.addWidget(killer_combo_box, 1, 2, 1, 3)
        grid.addWidget(qt.QLabel("Killer:"), 1, 1, 1, 1)
        
        assist_combo_boxes = [None] * 5
        for i in range(5):
            assist_combo_boxes[i] = qt.QComboBox()
            assist_combo_boxes[i].addItem("")
            assist_combo_boxes[i].addItems(current_player_heroes)
            grid.addWidget(assist_combo_boxes[i], 2 + i, 2, 1, 3)
            grid.addWidget(qt.QLabel("Assist %d:" % (i + 1)), 2 + i, 1, 1, 1)
        
        ability_combo_box = qt.QComboBox()
        grid.addWidget(ability_combo_box, 7, 2, 1, 2)
        grid.addWidget(qt.QLabel("Ability:"), 7, 1, 1, 1)
        
        critical_check_box = qt.QCheckBox("Critical")
        grid.addWidget(critical_check_box, 7, 4, 1, 1)
        
        def update_abilities(killer):
            ability_combo_box.setCurrentIndex(0)
            ability_combo_box.clear()
            ability_combo_box.addItem("KILL")
            
            if killer == "":
                return
            
            killer_hero_name = killer.split(" - ")[1]
            for hero in self.__heroes:
                if hero.name == killer_hero_name:
                    killer_hero = hero
                    break
            
            for ability in killer_hero.abilities:
                ability_combo_box.addItem(ability.name)
        
        killee_combo_box = qt.QComboBox()
        killee_combo_box.addItem("")
        killee_combo_box.addItems(itertools.chain(*[
            [
                (player_name + " - " + hero_alt) for hero_alt in hero_alts
            ] for player_name, hero_alts in [
                (player_name, [hero.name, *[(hero.name + "-" + alt) for alt in hero.alts]]) for player_name, hero in [
                    (player["name"], self.__lookup_hero(player["heroes"].current(current_time)["name"])) for player in current_match["players"]
                ]
            ]
        ]))
        grid.addWidget(killee_combo_box, 8, 2, 1, 3)
        grid.addWidget(qt.QLabel("Killee: "), 8, 1, 1, 1)
        
        if edit is not None and isinstance(edit, int):
            editing_kill = current_match["kills"][edit]
            if editing_kill["killer"] is not None:
                killer_combo_box.setCurrentText(editing_kill["killer"]["name"] + " - " + editing_kill["killer"]["hero"])
                update_abilities(killer_combo_box.currentText())
                ability_combo_box.setCurrentText(editing_kill["ability"])
                try:
                    if editing_kill["critical"]:
                        critical_check_box.setCheckState(2)
                except KeyError:
                    pass
            for i in range(min(len(assist_combo_boxes), len(editing_kill["assists"]))):
                assist_combo_boxes[i].setCurrentText(editing_kill["assists"][i]["name"] + " - " + editing_kill["assists"][i]["hero"])
            killee_combo_box.setCurrentText(editing_kill["killee"]["name"] + " - " + editing_kill["killee"]["hero"])
        else:
            update_abilities(killer_combo_box.currentText())
        
        killer_combo_box.currentTextChanged.connect(update_abilities)
        
        def add_kill():
            killee_name = killee_combo_box.currentText()
            if killee_name == "":
                self.__show_warning("Killee must be selected")
                return
            
            killee_name, killee_hero = killee_name.split(" - ")
            
            killer_index = killer_combo_box.currentIndex() - 1
            if killer_index < 0:
                killer_index = None
            
            if ability_combo_box.currentIndex() == 0:
                ability = None
            else:
                ability = ability_combo_box.currentText()
            
            assists = []
            for assist_combo_box in assist_combo_boxes:
                assist_index = assist_combo_box.currentIndex() - 1
                if assist_index < 0:
                    continue
                
                assists.append(assist_index)
            
            kill_params = (
                (killee_name, killee_hero),
                killer_index,
                assists,
                ability,
                critical_check_box.checkState() > 0
            )
            if edit is None:
                self.__descriptor.add_kill(
                    current_time,
                    *kill_params,
                )
            else:
                self.__descriptor.update_kill(
                    current_time,
                    edit,
                    *kill_params,
                    current_match=current_match,
                )
            
            widget.close()
        
        ok_btn = qt.QPushButton("Add Kill" if edit is None else "Save Kill")
        ok_btn.clicked.connect(add_kill)
        grid.addWidget(ok_btn, 9, 3, 1, 2)
        
        widget.show()
        
    def __open_remove_kill_window(self):
        current_time = self.__current_time()
        try:
            current_match = self.__descriptor.current_match(current_time)
        except ValueError as e:
            self.__show_warning(str(e))
            return
        
        match_all_kills = [
            (_format_time(kill["start_time"]) + ": " + _format_kill(kill)) for kill in current_match["kills"]
        ]
        
        widget = self.__create_window("Remove Kill")
        
        grid = qt.QGridLayout(widget)
        
        kill_combo_box = qt.QComboBox()
        kill_combo_box.addItem("")
        kill_combo_box.addItems(match_all_kills)
        grid.addWidget(kill_combo_box, 1, 1, 1, 2)
        
        def edit_kill():
            kill_index = kill_combo_box.currentIndex() - 1
            if kill_index < 0:
                return
            
            widget.close()
            self.__open_add_kill_window(edit=kill_index)
        
        ok_btn = qt.QPushButton("Edit")
        ok_btn.clicked.connect(edit_kill)
        grid.addWidget(ok_btn, 2, 1, 1, 1)
        
        def remove_kill():
            kill_index = kill_combo_box.currentIndex() - 1
            if kill_index < 0:
                return
            
            del current_match["kills"][kill_index]
            self.__descriptor.set_updated()
            
            widget.close()
        
        ok_btn = qt.QPushButton("Remove")
        ok_btn.clicked.connect(remove_kill)
        grid.addWidget(ok_btn, 2, 2, 1, 1)
        
        widget.show()
    
    def __handle_double_click(self, x, y, flags):
        for i, player_hero_point in enumerate(self.__player_hero_points):
            if player_hero_point[0] < x < player_hero_point[0] + self.__player_hero_size[0] and player_hero_point[1] < y < player_hero_point[1] + self.__player_hero_size[1]:
                self.__edit_player_hero(i)
                return

    def __should_render(self):
        if self.__descriptor.updated():
            return True
        
        if self.__state[_Visualizer.__STATE__MESSAGE] != self.__message:
            return True
        
        if self.__state[_Visualizer.__STATE__OVERLAY] != self.__hide_overlay:
            return True
        
        if self.__state[_Visualizer.__STATE__CURSOR] != self.__cursor:
            return True
        
        if self.__state[_Visualizer.__STATE__EXTRACTION_PROGRESS] != self.__extraction_progress:
            return True
        
        return False
    
    def __rollback(self):
        self.__message = self.__state[_Visualizer.__STATE__MESSAGE]
        self.__hide_overlay = self.__state[_Visualizer.__STATE__OVERLAY]
        self.__cursor = self.__state[_Visualizer.__STATE__CURSOR]
        self.__extraction_progress = self.__state[_Visualizer.__STATE__EXTRACTION_PROGRESS]
    
    def __update_state(self):
        self.__state = {
            _Visualizer.__STATE__MESSAGE: self.__message,
            _Visualizer.__STATE__OVERLAY: self.__hide_overlay,
            _Visualizer.__STATE__CURSOR: self.__cursor,
            _Visualizer.__STATE__EXTRACTION_PROGRESS: self.__extraction_progress,
        }
    
    def __current_time(self):
        return self.__cursor * self.__metadata.frame_interval / self.__metadata.frame_rate
    
    def __total_time(self):
        return self.__metadata.frame_count / self.__metadata.frame_rate
    
    def __draw_text(self, img, **kwargs):
        h, w, _ = img.shape
        
        if self.__extraction_progress is not None:
            img = _draw_text(
                img,
                "extracting %.2f%%" % (float(self.__extraction_progress[0]) / float(self.__extraction_progress[1]) * 100.,),
                (0, h),
                self.__text_font,
                **kwargs,
            )
            
        img = _draw_text(
            img,
            _format_time(self.__current_time()) + "/" + _format_time(self.__total_time()),
            (w, h),
            self.__text_font,
            halign="right",
            **kwargs,
        )
        
        labels = self.__descriptor.instantaneous_labels(self.__current_time())
        img = _draw_text(
            img,
            labels["match"],
            (w//2, 0),
            self.__text_font,
            valign="top",
            halign="center",
            **kwargs,
        )
        
        for i, player_name in enumerate(labels["player_names"]):
            img = _draw_text(
                img,
                player_name,
                (self.__player_hero_points[i][0] + self.__player_hero_size[0]//2, self.__player_hero_points[i][1] - 1.j),
                self.__text_font,
                valign="top",
                halign="center",
                **kwargs,
            )
        
        for i, player_hero in enumerate(labels["player_heroes"]):
            img = _draw_text(
                img,
                player_hero,
                (self.__player_hero_points[i][0] + self.__player_hero_size[0]//2, self.__player_hero_points[i][1]),
                self.__text_font,
                valign="top",
                halign="center",
                **kwargs,
            )
        
        for i, kill_str in enumerate(labels["kills"]):
            img = _draw_text(
                img,
                kill_str,
                (self.__kill_feed_pos[0], self.__kill_feed_pos[1] + (i * 1j)),
                self.__text_font,
                valign="top",
                halign="left",
                **kwargs,
            )
        
        if self.__message is not None:
            img = _draw_text(
                img,
                self.__message,
                (w//2, h),
                self.__text_font,
                halign="center",
                **kwargs,
            )
        
        return img
    
    def __render_state(self):
        if self.__render_cache is None or self.__should_render():
            img = None
            try:
                with open(self.__frames[self.__cursor], 'rb') as r:
                    buf = np.fromstring(r.read(), dtype='uint8')
                if len(buf) > 0:
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except FileNotFoundError:
                pass
            
            if img is None:
                self.__rollback()
                self.__message = "please wait for extraction"
                return None
            
            self.__render_cache = img
            if self.__hide_overlay:
                self.__update_state()
                return self.__render_cache
            
            # draw boxes around player heroes
            if self.__draw_boxes:
                for player_hero_point in self.__player_hero_points:
                   self.__render_cache = cv2.rectangle(self.__render_cache, player_hero_point, (player_hero_point[0] + self.__player_hero_size[0], player_hero_point[1] + self.__player_hero_size[1]), (0, 0, 255), 2)
                
            self.__render_cache = self.__draw_text(self.__render_cache)

            self.__update_state()
            
            return self.__render_cache
        
        return None
    
    def __wait(self, t=1):
        key = cv2.waitKeyEx(t)
        if key != -1:
            self.__message = None
        
        if key == 27: # ESC
            self.__hide_overlay = not self.__hide_overlay
            
        elif key == 32: # SPACE
            self.__open_match_buttons()
            
        elif key == 115: # s
            self.__message = "saved"
            self.__descriptor.save()
            
        elif key == 100: # d
            self.__open_remove_kill_window()
            
        elif key == 97: # a
            self.__open_add_kill_window()
            
        elif key == 49: # 1
            self.__edit_player_hero(0)
            
        elif key == 50: # 2
            self.__edit_player_hero(1)
            
        elif key == 51: # 3
            self.__edit_player_hero(2)
            
        elif key == 52: # 4
            self.__edit_player_hero(3)
            
        elif key == 53: # 5
            self.__edit_player_hero(4)
            
        elif key == 54: # 6
            self.__edit_player_hero(5)
            
        elif key == 55: # 7
            self.__edit_player_hero(6)
            
        elif key == 56: # 8
            self.__edit_player_hero(7)
            
        elif key == 57: # 9
            self.__edit_player_hero(8)
            
        elif key == 48: # 0
            self.__edit_player_hero(9)
            
        elif key == 45: # -
            self.__edit_player_hero(10)
            
        elif key == 61: # =
            self.__edit_player_hero(11)
        
        elif key != -1:
            print("unbound key:", key, bin(key), "(" + chr(key%256) + ")")
        
        if cv2.getWindowProperty(self.__window_name, cv2.WND_PROP_ASPECT_RATIO) < 0: # exited
            return False
        
        return True
    
    def loop(self):
        while True:
            frame = self.__render_state()
            if frame is not None:
                cv2.imshow(self.__window_name, frame)
            
            if not self.__wait():
                break


def visualize(*args, **kwargs):
    with _Visualizer(*args, **kwargs) as v:
        v.loop()


def _parse_point(str):
    return tuple([int(i) for i in str.split(",")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--dest", dest="dest", default=None, required=False)
    parser.add_argument("--framerate", dest="target_framerate", default=10., type=float)
    parser.add_argument("--size", dest="target_size", type=_size, default=(1280, 720))
    parser.add_argument("--description", dest="description_def", default=None)
    parser.add_argument("--config", dest="config_file", default="config.ini")
    parser.add_argument("--show-boxes", dest="draw_boxes", const=True, nargs='?', default=False, type=bool)
    args = parser.parse_args()
    
    if args.description_def is None:
        args.description_def = os.path.splitext(args.video)[0] + ".description.json"
    
    try:
        visualize(
            _VisualizerConfig(args.config_file),
            args.video,
            _Descriptor(args.description_def),
            draw_boxes=args.draw_boxes,
        )
    except KeyboardInterrupt:
        pass
