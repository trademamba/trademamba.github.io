<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


# Call Rolling Algo Version 1
For a short call expiring today $C$ that is in the money we find the value of $C_1(S),\dots C_n(S)$ the value of rolling the call to the next $1,\dots,n$ strikes. For a value $\Delta S$ for the change in the stock depending on some sort of volatility or config we calculate the expected value of $C_1(S-\DeltaS),\dots C_n(S-\DeltaS)$.


# To Dos

1. Make sure you don't get wiped out with expiring bull calls (i.e. 162.5/165 bull call spreads)
  * Ensure you have margin to buy if 165 worthless and 162.5 is in the money
  * As soon as price gets close to 160 then do either buy 100 shares sell 1 162.5 call or buy 1 157.5 call and sell 1 160 call and keep going... That said wait for the last possible second to do this
