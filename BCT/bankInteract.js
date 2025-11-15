// scripts/interact.js
async function main() {
  const [user] = await ethers.getSigners();
  const bankAddress = "<PASTE_DEPLOYED_ADDRESS_HERE>";
  const Bank = await ethers.getContractFactory("Bank");
  const bank = Bank.attach(bankAddress);

  // Deposit 0.01 ETH
  const depositTx = await bank.connect(user).deposit({ value: ethers.utils.parseEther("0.01") });
  const depRec = await depositTx.wait();
  console.log("Deposit gasUsed:", depRec.gasUsed.toString());

  // Show balance (caller)
  const bal = await bank.connect(user).balance();
  console.log("Balance (wei):", bal.toString());
  console.log("Balance (ETH):", ethers.utils.formatEther(bal));

  // Withdraw 0.005 ETH
  const withdrawAmount = ethers.utils.parseEther("0.005");
  const wdTx = await bank.connect(user).withdraw(withdrawAmount);
  const wdRec = await wdTx.wait();
  console.log("Withdraw gasUsed:", wdRec.gasUsed.toString());

  // Balance after withdraw
  const bal2 = await bank.connect(user).balance();
  console.log("Balance after withdraw (ETH):", ethers.utils.formatEther(bal2));
}
main().catch((err) => { console.error(err); process.exitCode = 1; });
// scripts/interact.js
async function main() {
  const [user] = await ethers.getSigners();
  const bankAddress = "<PASTE_DEPLOYED_ADDRESS_HERE>";
  const Bank = await ethers.getContractFactory("Bank");
  const bank = Bank.attach(bankAddress);

  // Deposit 0.01 ETH
  const depositTx = await bank.connect(user).deposit({ value: ethers.utils.parseEther("0.01") });
  const depRec = await depositTx.wait();
  console.log("Deposit gasUsed:", depRec.gasUsed.toString());

  // Show balance (caller)
  const bal = await bank.connect(user).balance();
  console.log("Balance (wei):", bal.toString());
  console.log("Balance (ETH):", ethers.utils.formatEther(bal));

  // Withdraw 0.005 ETH
  const withdrawAmount = ethers.utils.parseEther("0.005");
  const wdTx = await bank.connect(user).withdraw(withdrawAmount);
  const wdRec = await wdTx.wait();
  console.log("Withdraw gasUsed:", wdRec.gasUsed.toString());

  // Balance after withdraw
  const bal2 = await bank.connect(user).balance();
  console.log("Balance after withdraw (ETH):", ethers.utils.formatEther(bal2));
}
main().catch((err) => { console.error(err); process.exitCode = 1; });


// Run interaction (after replacing bankAddress):

// npx hardhat run --network <your-testnet> scripts/interact.js