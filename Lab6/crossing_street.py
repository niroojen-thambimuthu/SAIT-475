# %%
from durable.lang import *

with ruleset("crossing_street"):

    @when_all(m.color == "green")
    def green_light(c):
        print("Go.")

    # yellow light?
    @when_all(m.color == "yellow")
    def yellow_light(c):
        print("Slow down.")

    # red light? 
    @when_all(m.color == "red")
    def red_light(c):
        print("Stop.")

post("crossing_street", {"color": "red"})
post("crossing_street", {"color": "green"})
post("crossing_street", {"color": "yellow"})
