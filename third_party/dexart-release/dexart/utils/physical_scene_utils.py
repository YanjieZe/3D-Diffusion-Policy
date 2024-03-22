from typing import List

import numpy as np
from sapien import core as sapien


def is_link_group_contact(scene: sapien.Scene, actors1: List[sapien.Actor], actors2: List[sapien.Actor]) -> bool:
    actor_set1 = set(actors1)
    actor_set2 = set(actors2)
    for contact in scene.get_contacts():
        contact_actors = {contact.actor0, contact.actor1}
        if len(actor_set1 & contact_actors) > 0 and len(actor_set2 & contact_actors) > 0:
            impulse = [point.impulse for point in contact.points]
            if np.sum(np.abs(impulse)) < 1e-6:
                continue
            return True
    return False


def get_unique_contact(scene: sapien.Scene):
    contact_list = []
    contact_actor_list = []
    for contact in scene.get_contacts():
        impulse = [point.impulse for point in contact.points]
        if np.sum(np.abs(impulse)) < 1e-6:
            continue

        if contact.actor0.get_id() < contact.actor1.get_id():
            contact_actor = (contact.actor0, contact.actor1)
        else:
            contact_actor = (contact.actor1, contact.actor0)
        if contact_actor in contact_list:
            continue

        contact_actor_list.append(contact_actor)
        contact_list.append(contact)

    return contact_actor_list, contact_list
