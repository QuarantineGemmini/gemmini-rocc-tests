- __uart-printf notes__
  - i explored using a hacked UART (250 MHz) to do printfs directly from 
    software instead of using the TSI-based HTIF approach. I thought that 
    the HTIF approach is not bare-metal and is slow.
  - i was wrong on the second point... its actually faster to simulate a 
    bunch of software printfs by using TSI.
  - the uart files are here for reference
