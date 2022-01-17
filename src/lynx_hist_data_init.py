import sys
import lynx_hist_data
import common


if __name__ == "__main__":
    if common.already_running():
        exit()

    if common.already_finished():
        exit()

    common.log("Start script")
    start_conid = None
    for x in sys.argv:
        if x.startswith("--start_conid="):
            start_conid = x.split("=")[1]
        
    common.log_debug("start_conid:", start_conid)
    
    lynx_hist_data.main(period="15y", append=False, start_conid=start_conid, if_not_exists=True, remove_incomplete=True)
    common.log("Finish script")
