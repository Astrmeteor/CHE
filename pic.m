% X = randi([10,1000],1,500);
X = 10:1:500
Y = 10:1:500
% Y = randi([10,1000],1,500);
[X,Y]=meshgrid(X,Y);
Z = (X+Y-2)./(2.*X-2);
i = find(X<Y & Z>1);
Z(i)=NaN;
% surf(X,Y,Z, 'FaceColor',[0 1 0]);
surf(X,Y,Z,'EdgeColor','none');
xlabel('Input size');
ylabel('Output size');
zlabel('Ratio');
