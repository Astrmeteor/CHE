yyaxis left
y1 = [10.2 39.78;0 0];
b1 = bar(y1,0.5,'grouped');
b1(1).FaceColor='blue';
b1(2).FaceColor='red';
ylim([0 100]);
ylabel('Pixel-wise Decryption Time(s)');

hold on;
yyaxis right
y2 = [0 0; 0.01 0.03];
b2 = bar(y2,0.5,'grouped');
b2(1).FaceColor='blue';
b2(2).FaceColor='red';
ylim([0 1]);
ylabel('Channel-wise Decryption Time(s)');

set(gca,'XTickLabel',{'Pixel-wise Encryption','Channel-wise Encryptpion'})
legend('MNIST','CIFAR-10');
xlabel('Encryption Method');

xtips1 = b1.XEndPoints;
ytip1 = b1.YEndPoints;
labels1 = string(b1.YData);
text(xtips1, ytip1, labels1, "HorizontalAlignment","center",...
    "VerticalAlignment","bottom");