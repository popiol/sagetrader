import sys
import lynx_hist_data


if __name__ == "__main__":
    start_conid = None
    for x in sys.argv:
        if x.startswith("--start_conid="):
            start_conid = x.split("=")[1]
        
    lynx_hist_data.main(period="15y", append=False, start_conid=start_conid)
