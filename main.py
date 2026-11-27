from gui import parse_config,GUI


config = parse_config("config.txt")
gui = GUI(config)
gui.run()



